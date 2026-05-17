"""
PP-FormulaNet Optimized Decoder ONNX Export (KV-Cache)
=======================================================
Token-only output (argmax, 8 bytes/step) + KV-cache for O(n) generation.

Decoder variants:
  - Prefill: first step (BOS + enc) -> token_id + 32 KV
  - Decode:  incremental (self-KV + enc) -> token_id + 32 KV
            (cross-KV recomputed from encoder each step — matches m.generate())
  - Merged:  first and incremental steps in one graph
             (zero-length self-KV on the first step, outputs token_id + 16 self-KV)

Usage:
    python run_conversion_optimized.py                       # Batch over imgs/
    python run_conversion_optimized.py --model plus          # Plus-L only
    python run_conversion_optimized.py --image one.png       # Single image
    python run_conversion_optimized.py --skip-export         # Verify existing
"""

import os, sys, json, time, argparse
from pathlib import Path
import torch, torch.nn as nn, numpy as np
from PIL import Image
from transformers.models.pp_formulanet import PPFormulaNetForConditionalGeneration
from transformers import AutoProcessor
from transformers.cache_utils import EncoderDecoderCache, DynamicCache, DynamicLayer
import onnxruntime as ort

# ============================================================================
# Patches: mask functions + DynamicLayer for JIT compatibility
# ============================================================================
def _apply_patches():
    import transformers.models.pp_formulanet.modeling_pp_formulanet as pp
    pp.create_causal_mask = lambda *a,**k: None
    pp.create_bidirectional_mask = lambda *a,**k: None

    # Patch DynamicLayer: no empty tensors, explicit is_initialized
    def dl_init(self, *a, **k):
        self.is_initialized = False

    def dl_update(self, key, value):
        if not self.is_initialized:
            self.keys = key.new_zeros(0)
            self.values = value.new_zeros(0)
            self.is_initialized = True
        self.keys = torch.cat([self.keys, key], dim=2)
        self.values = torch.cat([self.values, value], dim=2)
        return self.keys, self.values

    def dc_update(self, key, value, layer_idx):
        if self.layers is None: self.layers = []
        if len(self.layers) <= layer_idx:
            l = DynamicLayer(); l.keys = key; l.values = value; l.is_initialized = True
            self.layers.append(l)
        else:
            self.layers[layer_idx].update(key, value)
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    DynamicLayer.__init__ = dl_init
    DynamicLayer.update = dl_update
    DynamicCache.update = dc_update

_apply_patches()

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = str(BASE_DIR / "imgs")
IMG = DEFAULT_INPUT
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
N = 8  # decoder layers
MODEL_CONFIGS = {
    "PP-FormulaNet_plus-L": {
        "model_dir": r"D:\models\PP-FormulaNet_plus-L_safetensors",
        "onnx_encoder": "onnx/PP-FormulaNet_plus-L_encoder.onnx",
        "onnx_prefill": "onnx/PP-FormulaNet_plus-L_decoder_prefill.onnx",
        "onnx_decode":  "onnx/PP-FormulaNet_plus-L_decoder_decode.onnx",
        "onnx_merged":  "onnx/PP-FormulaNet_plus-L_decoder_model_merged.onnx",
    },
    "PP-FormulaNet-L": {
        "model_dir": r"D:\models\PP-FormulaNet-L_safetensors",
        "onnx_encoder": "onnx/PP-FormulaNet-L_encoder.onnx",
        "onnx_prefill": "onnx/PP-FormulaNet-L_decoder_prefill.onnx",
        "onnx_decode":  "onnx/PP-FormulaNet-L_decoder_decode.onnx",
        "onnx_merged":  "onnx/PP-FormulaNet-L_decoder_model_merged.onnx",
    },
}
OUT = Path("onnx")

# ============================================================================
# KV helpers
# ============================================================================
def kv_names(pref, n=N):
    r = []
    for i in range(n):
        for s in ['sk', 'sv', 'ck', 'cv']:
            r.append(f'{pref}_{i}_{s}')
    return r

def kv_names_self(pref, n=N):
    """Self-attention KV names only (sk, sv per layer)."""
    r = []
    for i in range(n):
        for s in ['sk', 'sv']:
            r.append(f'{pref}_{i}_{s}')
    return r

def kv_daxes(names, sd='S', cd='C'):
    d = {}
    for n in names:
        d[n] = {0: 'B', 2: sd if n.split('_')[-1] in ('sk', 'sv') else cd}
    return d

def flat(pk):
    r = []
    for i in range(len(pk.self_attention_cache.layers)):
        r.append(pk.self_attention_cache.layers[i].keys)
        r.append(pk.self_attention_cache.layers[i].values)
        r.append(pk.cross_attention_cache.layers[i].keys)
        r.append(pk.cross_attention_cache.layers[i].values)
    return r

def resolve_input_paths(input_path):
    p = Path(input_path)
    if not p.exists():
        alt = BASE_DIR / p
        if alt.exists():
            p = alt
    if not p.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if p.is_dir():
        paths = sorted(
            [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS],
            key=lambda x: x.name.lower(),
        )
    else:
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image type: {p}")
        paths = [p]
    if not paths:
        raise FileNotFoundError(f"no supported images found in: {p}")
    return paths

def load_images(paths, proc):
    imgs = []
    for path in paths:
        with Image.open(path) as im:
            imgs.append(im.convert("RGB"))
    return proc(images=imgs, return_tensors="pt")["pixel_values"]

def extract_self_kv(outputs):
    pkv_all = list(outputs[1:])
    pkv_self = []
    for layer in range(N):
        base = layer * 4
        pkv_self.append(pkv_all[base])     # self-key
        pkv_self.append(pkv_all[base + 1]) # self-value
    return pkv_self

def token_list(t):
    if isinstance(t, torch.Tensor):
        return t.reshape(-1).tolist()
    return np.asarray(t).reshape(-1).tolist()

def format_preview(text, limit=65536):
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 3] + "..."

def generation_max_length(m):
    mx = getattr(m.generation_config, 'max_length', None)
    if mx is None:
        mx = getattr(m.config, 'max_length', None)
    if mx is None:
        mx = getattr(m.config.text_config, 'max_length', None)
    if mx is None:
        raise ValueError("model generation max_length is not configured")
    return int(mx)

# ============================================================================
# Wrappers for ONNX export
# ============================================================================

class PrefillW(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.d = m.model.decoder; self.l = m.lm_head

    def forward(self, ids, enc):
        o = self.d(input_ids=ids, encoder_hidden_states=enc, use_cache=True)
        h = o.last_hidden_state[:, -1:, :]
        t = torch.argmax(self.l(h), dim=-1, keepdim=True)
        return (t,) + tuple(flat(o.past_key_values))


class DecodeW(nn.Module):
    """Decode step: ids + enc + 16 self-KV -> token_id + 32 KV (self+cross).
    Cross-KV is NOT cached between steps — decoder recomputes from encoder_hidden_states.
    This matches m.generate() behavior and forces enc in the ONNX graph."""
    def __init__(self, m):
        super().__init__()
        self.d = m.model.decoder; self.l = m.lm_head
        self.c = m.model.decoder.config

    def forward(self, ids, enc,
                sk0, sv0, sk1, sv1, sk2, sv2, sk3, sv3,
                sk4, sv4, sk5, sv5, sk6, sv6, sk7, sv7):
        sa = DynamicCache(config=self.c); ca = DynamicCache(config=self.c)
        sa.layers = []
        # ca.layers stays None — forces cross-attention to use encoder_hidden_states
        for idx, (sk, sv) in enumerate([
            (sk0, sv0), (sk1, sv1), (sk2, sv2), (sk3, sv3),
            (sk4, sv4), (sk5, sv5), (sk6, sv6), (sk7, sv7)]):
            sl = DynamicLayer(); sl.keys = sk; sl.values = sv; sl.is_initialized = True
            sa.layers.append(sl)
        o = self.d(input_ids=ids, encoder_hidden_states=enc,
                    past_key_values=EncoderDecoderCache(sa, ca), use_cache=True)
        h = o.last_hidden_state[:, -1:, :]
        t = torch.argmax(self.l(h), dim=-1, keepdim=True)
        so = o.past_key_values.self_attention_cache
        co = o.past_key_values.cross_attention_cache
        return (t,
            so.layers[0].keys, so.layers[0].values, co.layers[0].keys, co.layers[0].values,
            so.layers[1].keys, so.layers[1].values, co.layers[1].keys, co.layers[1].values,
            so.layers[2].keys, so.layers[2].values, co.layers[2].keys, co.layers[2].values,
            so.layers[3].keys, so.layers[3].values, co.layers[3].keys, co.layers[3].values,
            so.layers[4].keys, so.layers[4].values, co.layers[4].keys, co.layers[4].values,
            so.layers[5].keys, so.layers[5].values, co.layers[5].keys, co.layers[5].values,
            so.layers[6].keys, so.layers[6].values, co.layers[6].keys, co.layers[6].values,
            so.layers[7].keys, so.layers[7].values, co.layers[7].keys, co.layers[7].values)


class MergedDecoderW(nn.Module):
    """Merged prefill/decode step.

    The first step passes zero-length self-KV; later steps pass the previous
    self-KV outputs. Cross-KV is recomputed from encoder_hidden_states and is
    intentionally not returned.
    """
    def __init__(self, m):
        super().__init__()
        self.d = m.model.decoder; self.l = m.lm_head
        self.c = m.model.decoder.config

    def forward(self, ids, enc,
                sk0, sv0, sk1, sv1, sk2, sv2, sk3, sv3,
                sk4, sv4, sk5, sv5, sk6, sv6, sk7, sv7):
        sa = DynamicCache(config=self.c); ca = DynamicCache(config=self.c)
        sa.layers = []
        # ca.layers stays None so cross-attention is recomputed from enc.
        for idx, (sk, sv) in enumerate([
            (sk0, sv0), (sk1, sv1), (sk2, sv2), (sk3, sv3),
            (sk4, sv4), (sk5, sv5), (sk6, sv6), (sk7, sv7)]):
            sl = DynamicLayer(); sl.keys = sk; sl.values = sv; sl.is_initialized = True
            sa.layers.append(sl)
        o = self.d(input_ids=ids, encoder_hidden_states=enc,
                    past_key_values=EncoderDecoderCache(sa, ca), use_cache=True)
        h = o.last_hidden_state[:, -1:, :]
        t = torch.argmax(self.l(h), dim=-1, keepdim=True)
        so = o.past_key_values.self_attention_cache
        return (t,
            so.layers[0].keys, so.layers[0].values,
            so.layers[1].keys, so.layers[1].values,
            so.layers[2].keys, so.layers[2].values,
            so.layers[3].keys, so.layers[3].values,
            so.layers[4].keys, so.layers[4].values,
            so.layers[5].keys, so.layers[5].values,
            so.layers[6].keys, so.layers[6].values,
            so.layers[7].keys, so.layers[7].values)


# ============================================================================
# Export
# ============================================================================
def exp_enc(m, p, pv):
    print(f"  Encoder -> {p}")
    class W(nn.Module):
        def __init__(self, _m): super().__init__(); self.e = _m.model.encoder
        def forward(self, x): return self.e(x).pooler_output
    w = W(m); w.eval()
    with torch.no_grad(): print(f"    out: {w(pv).shape}")
    torch.onnx.export(w, (pv,), p, input_names=['pv'], output_names=['enc'],
        dynamic_axes={'pv': {0: 'B'}, 'enc': {0: 'B'}}, opset_version=17, do_constant_folding=True)
    print("    OK")

def exp_pre(m, p, ids, enc):
    print(f"  Prefill -> {p}")
    w = PrefillW(m); w.eval()
    with torch.no_grad(): o = w(ids, enc); print(f"    tok:{o[0].shape} kv:{len(o)-1}")
    in_ = ['ids', 'enc']; on_ = ['tok'] + kv_names('pres')
    da = {'ids': {0: 'B', 1: 'T'}, 'enc': {0: 'B', 1: 'E'}, 'tok': {0: 'B'}}
    da.update(kv_daxes(kv_names('pres'), 'S', 'C'))
    torch.onnx.export(w, (ids, enc), p, input_names=in_, output_names=on_,
        dynamic_axes=da, opset_version=17, do_constant_folding=True)
    print("    OK")

def exp_dec(m, p, ids, enc, pk_flat):
    """Export decode decoder. pk_flat has 32 tensors; only self-KV (16) are used as input."""
    print(f"  Decode -> {p}")
    w = DecodeW(m); w.eval()
    # Extract self-KV only: sk at indices 0,4,8,...; sv at 1,5,9,...
    pk_self = []
    for layer in range(N):
        pk_self.append(pk_flat[layer*4])     # sk
        pk_self.append(pk_flat[layer*4+1])   # sv
    a = (ids, enc) + tuple(pk_self)
    with torch.no_grad(): o = w(*a); print(f"    tok:{o[0].shape} selfK:{o[1].shape[2]}")
    in_ = ['ids', 'enc'] + kv_names_self('past')
    on_ = ['tok'] + kv_names('pres')  # outputs: tok + 32 KV (self+cross)
    da = {'ids': {0: 'B', 1: 'T'}, 'enc': {0: 'B', 1: 'E'}, 'tok': {0: 'B'}}
    da.update(kv_daxes(kv_names_self('past'), 'pS', 'pS'))
    da.update(kv_daxes(kv_names('pres'), 'qS', 'C'))
    torch.onnx.export(w, a, p, input_names=in_, output_names=on_,
        dynamic_axes=da, opset_version=17, do_constant_folding=True)
    print("    OK")

def exp_merged(m, p, ids, enc, pk_flat):
    """Export one decoder graph for prefill and decode.

    The sample uses non-empty self-KV so the tracer records the cache append
    path. The resulting graph still accepts zero-length self-KV for the first
    decoding step because the KV length axis is dynamic.
    """
    print(f"  Merged decoder -> {p}")
    w = MergedDecoderW(m); w.eval()
    pk_self = []
    for layer in range(N):
        pk_self.append(pk_flat[layer*4])
        pk_self.append(pk_flat[layer*4+1])
    a = (ids, enc) + tuple(pk_self)
    with torch.no_grad(): o = w(*a); print(f"    tok:{o[0].shape} selfK:{o[1].shape[2]} kv:{len(o)-1}")
    in_ = ['ids', 'enc'] + kv_names_self('past')
    on_ = ['tok'] + kv_names_self('pres')
    da = {'ids': {0: 'B', 1: 'T'}, 'enc': {0: 'B', 1: 'E'}, 'tok': {0: 'B'}}
    da.update(kv_daxes(kv_names_self('past'), 'pS', 'pS'))
    da.update(kv_daxes(kv_names_self('pres'), 'qS', 'qS'))
    torch.onnx.export(w, a, p, input_names=in_, output_names=on_,
        dynamic_axes=da, opset_version=17, do_constant_folding=True)
    print("    OK")

# ============================================================================
# Inference
# ============================================================================
def load_m(d):
    print(f"  Load: {d}")
    m = PPFormulaNetForConditionalGeneration.from_pretrained(d, dtype=torch.float32); m.eval()
    p = AutoProcessor.from_pretrained(d)
    return m, p

def pt_gen(m, pv, sid, eid, mx):
    """Manual step-by-step PT decode matching ONNX logic for a batch."""
    with torch.no_grad():
        enc = m.model.encoder(pv).pooler_output
        bsz = pv.shape[0]
        cur = torch.full((bsz, 1), sid, dtype=torch.long, device=pv.device)
        o = m.model.decoder(input_ids=cur, encoder_hidden_states=enc, use_cache=True)
        pk = o.past_key_values
        t = torch.argmax(m.lm_head(o.last_hidden_state[:, -1:, :]), dim=-1)
        gen = [[sid, int(t[i].item())] for i in range(bsz)]
        finished = t.eq(eid)
        for _ in range(mx - 2):
            if bool(finished.all()):
                break
            nxt = torch.where(finished, torch.full_like(t, eid), t)
            o = m.model.decoder(input_ids=nxt, encoder_hidden_states=enc, past_key_values=pk, use_cache=True)
            pk = o.past_key_values
            t = torch.argmax(m.lm_head(o.last_hidden_state[:, -1:, :]), dim=-1)
            t = torch.where(finished, torch.full_like(t, eid), t)
            for i in range(bsz):
                if not bool(finished[i].item()):
                    gen[i].append(int(t[i].item()))
            finished = finished | t.eq(eid)
    return gen

def onx_gen(ep, pp, dp, pvn, sid, eid, mx):
    e = ort.InferenceSession(ep, providers=['CPUExecutionProvider'])
    pf = ort.InferenceSession(pp, providers=['CPUExecutionProvider'])
    dc = ort.InferenceSession(dp, providers=['CPUExecutionProvider'])
    enc = e.run(None, {'pv': pvn.astype(np.float32)})[0].astype(np.float32)
    bsz = pvn.shape[0]
    po = pf.run(None, {'ids': np.full((bsz, 1), sid, dtype=np.int64), 'enc': enc})
    t = po[0].reshape(-1).astype(np.int64)
    gen = [[sid, int(t[i])] for i in range(bsz)]
    # Prefill outputs 33 tensors (tok + 32 KV). Extract self-KV for decode input.
    pkv_self = extract_self_kv(po)
    kn_self = kv_names_self('past')
    finished = t == eid
    for _ in range(mx - 2):
        if bool(finished.all()):
            break
        cur = t.reshape(bsz, 1)
        cur = np.where(finished.reshape(bsz, 1), eid, cur).astype(np.int64)
        fd = {'ids': cur, 'enc': enc}
        for i, n in enumerate(kn_self): fd[n] = pkv_self[i]
        do = dc.run(None, fd)
        t = do[0].reshape(-1).astype(np.int64)
        t = np.where(finished, eid, t)
        for i in range(bsz):
            if not finished[i]:
                gen[i].append(int(t[i]))
        finished = finished | (t == eid)
        # Extract self-KV from decode output for next step
        pkv_self = extract_self_kv(do)
    return gen

def onx_gen_merged(ep, mp, pvn, sid, eid, mx):
    e = ort.InferenceSession(ep, providers=['CPUExecutionProvider'])
    dc = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
    enc = e.run(None, {'pv': pvn.astype(np.float32)})[0].astype(np.float32)
    bsz = pvn.shape[0]
    kn_self = kv_names_self('past')
    pkv_self = []
    for inp in dc.get_inputs()[2:]:
        shape = inp.shape
        heads = int(shape[1])
        head_dim = int(shape[3])
        pkv_self.append(np.zeros((bsz, heads, 0, head_dim), dtype=np.float32))

    cur = np.full((bsz, 1), sid, dtype=np.int64)
    gen = [[sid] for _ in range(bsz)]
    finished = np.zeros((bsz,), dtype=bool)
    for _ in range(mx - 1):
        fd = {'ids': cur, 'enc': enc}
        for i, n in enumerate(kn_self): fd[n] = pkv_self[i]
        out = dc.run(None, fd)
        t = out[0].reshape(-1).astype(np.int64)
        t = np.where(finished, eid, t)
        for i in range(bsz):
            if not finished[i]:
                gen[i].append(int(t[i]))
        finished = finished | (t == eid)
        pkv_self = list(out[1:])
        if bool(finished.all()):
            break
        cur = t.reshape(bsz, 1).astype(np.int64)
    return gen

# ============================================================================
# Pipeline
# ============================================================================
def run(name, cfg, args, image_paths):
    print(f"\n{'='*70}\n{name} (KV-CACHE)\n{'='*70}")
    OUT.mkdir(parents=True, exist_ok=True)
    ep = cfg['onnx_encoder']; pp = cfg['onnx_prefill']; dp = cfg['onnx_decode']; mp = cfg['onnx_merged']

    print("\n[1/4] Load...")
    m, proc = load_m(cfg['model_dir'])
    sid = m.generation_config.decoder_start_token_id
    eid = m.config.text_config.eos_token_id
    max_steps = generation_max_length(m)
    print(f"  start={sid} eos={eid} max_length={max_steps}")
    if args.steps is not None and args.steps != max_steps:
        print(f"  NOTE: --steps={args.steps} ignored; verification uses model max_length={max_steps}")

    print("\n[2/4] Images...")
    print(f"  batch size: {len(image_paths)}")
    for path in image_paths:
        print(f"  - {path.name}")
    pv = load_images(image_paths, proc)
    print(f"  shape: {pv.shape}")

    if not args.skip_export:
        print("\n[3/4] Export...")
        if not os.path.exists(ep) or args.force: exp_enc(m, ep, pv)
        else: print(f"  Encoder exists: {ep}")
        with torch.no_grad(): enc = m.model.encoder(pv).pooler_output
        bos = torch.full((pv.shape[0], 1), sid, dtype=torch.long)
        if not os.path.exists(pp) or args.force: exp_pre(m, pp, bos, enc)
        else: print(f"  Prefill exists: {pp}")
        pw = PrefillW(m); pw.eval()
        with torch.no_grad():
            po = pw(bos, enc); pk = list(po[1:])
            nt = po[0].reshape(pv.shape[0], -1).to(dtype=torch.long)
        if not os.path.exists(dp) or args.force:
            exp_dec(m, dp, nt, enc, pk)
        else: print(f"  Decode exists: {dp}")
        if not os.path.exists(mp) or args.force: exp_merged(m, mp, nt, enc, pk)
        else: print(f"  Merged decoder exists: {mp}")

    sl = '4' if not args.skip_export else '3'
    print(f"\n[{sl}/4] Verify merged decoder (max_length={max_steps})...")
    for x in [ep, mp]:
        if not os.path.exists(x): print(f"  ERROR: {x} missing"); return None

    print("  PT generate...")
    t0 = time.time(); pt = pt_gen(m, pv, sid, eid, max_steps); tp = time.time() - t0
    print(f"    batch complete in {tp:.1f}s")

    print("  ONNX generate (merged decoder)...")
    t0 = time.time(); ox = onx_gen_merged(ep, mp, pv.cpu().numpy(), sid, eid, max_steps); to = time.time() - t0
    print(f"    batch complete in {to:.1f}s")

    print(f"\n  {'='*60}\n  KV-CACHE RESULTS\n  {'='*60}")
    samples = []
    overall_ok = True
    for idx, path in enumerate(image_paths):
        pt_seq = pt[idx]
        ox_seq = ox[idx]
        compare_len = min(len(pt_seq), len(ox_seq))
        matched = sum(1 for i in range(compare_len) if pt_seq[i] == ox_seq[i])
        exact = len(pt_seq) == len(ox_seq) and matched == compare_len
        pt_text = proc.tokenizer.decode(pt_seq, skip_special_tokens=False)
        ox_text = proc.tokenizer.decode(ox_seq, skip_special_tokens=False)
        pt_eos = bool(pt_seq and pt_seq[-1] == eid)
        ox_eos = bool(ox_seq and ox_seq[-1] == eid)
        pt_trunc = len(pt_seq) >= max_steps and not pt_eos
        ox_trunc = len(ox_seq) >= max_steps and not ox_eos
        overall_ok = overall_ok and exact and (pt_text == ox_text)
        samples.append({
            'image': str(path),
            'image_name': path.name,
            'pt_tokens': pt_seq,
            'onnx_tokens': ox_seq,
            'pt_token_count': len(pt_seq),
            'onnx_token_count': len(ox_seq),
            'matched_prefix': matched,
            'compare_len': compare_len,
            'match_pct': 100 * matched / compare_len if compare_len > 0 else 0,
            'paths_match': exact,
            'texts_match': pt_text == ox_text,
            'pt_ended_with_eos': pt_eos,
            'onnx_ended_with_eos': ox_eos,
            'pt_truncated': pt_trunc,
            'onnx_truncated': ox_trunc,
            'pt_text': pt_text,
            'onnx_text': ox_text,
        })

        status = "OK" if exact and pt_text == ox_text else "FAIL"
        print(f"  {path.name}: {status} | tokens PT/ONNX={len(pt_seq)}/{len(ox_seq)} | eos PT/ONNX={pt_eos}/{ox_eos}")
        print(f"    text: {format_preview(pt_text)}")

    su = tp / to if to > 0 else 0
    print(f"\n  PT:{tp:.1f}s ONNX:{to:.1f}s Speedup:{su:.2f}x")
    print(f"  Batch: {len(image_paths)} image(s) | overall: {'OK' if overall_ok else 'FAIL'}")

    rpt = {
        'model': name,
        'decoder_variant': 'merged_self_kv',
        'input_source': args.image,
        'image_count': len(image_paths),
        'image_names': [p.name for p in image_paths],
        'max_steps': max_steps,
        'samples': samples,
        'overall_ok': overall_ok,
        'pt_time': tp,
        'onnx_time': to,
        'speedup': su,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report_path = OUT / f"{name}_kv_verification.json"
    with open(report_path, 'w') as f: json.dump(rpt, f, indent=2)
    print(f"  Report: {report_path}")
    return {'ok': overall_ok, 'pt': tp, 'onnx': to}

def main():
    pa = argparse.ArgumentParser(description="PP-FormulaNet KV-Cache ONNX Export")
    pa.add_argument('--model', default='all', choices=['all', 'plus', 'L'])
    pa.add_argument('--steps', type=int, default=None,
                    help='Deprecated; ignored. Verification always uses model generation_config.max_length.')
    pa.add_argument('--skip-export', action='store_true')
    pa.add_argument('--force', action='store_true')
    pa.add_argument('--image', default=IMG)
    a = pa.parse_args()
    try:
        image_paths = resolve_input_paths(a.image)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print("="*70+"\nPP-FormulaNet KV-Cache ONNX Export\n"+"="*70)
    print("Output: token_id (8B/step) + KV-cache O(n)")
    print(f"Input: {a.image}")
    print(f"Resolved images: {len(image_paths)}")
    rr = {}
    if a.model in ('all', 'plus'):
        rr['Plus-L'] = run('PP-FormulaNet_plus-L', MODEL_CONFIGS['PP-FormulaNet_plus-L'], a, image_paths)
    if a.model in ('all', 'L'):
        rr['L'] = run('PP-FormulaNet-L', MODEL_CONFIGS['PP-FormulaNet-L'], a, image_paths)
    print("\n"+"="*70+"\nSUMMARY\n"+"="*70)
    ao = True
    for k, r in rr.items():
        if r: s = "OK" if r['ok'] else "FAIL"; print(f"  {k}: {s} | PT:{r['pt']:.1f}s ONNX:{r['onnx']:.1f}s"); ao = ao and r['ok']
        else: print(f"  {k}: FAILED"); ao = False
    print("\n*** ALL PASSED! ***" if ao else "\n*** Some failed ***")
    sys.exit(0 if ao else 1)

if __name__ == '__main__': main()
