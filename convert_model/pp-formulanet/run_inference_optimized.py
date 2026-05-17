"""
PP-FormulaNet Optimized Inference (KV-Cache)
=============================================
Standalone inference using KV-cache ONNX decoders.
- Merged decoder: one ONNX model handles both first step and incremental decode
  by accepting zero-length self-KV on the first step.
- Split decoder compatibility: pass --prefill and --decode to use old models.

Usage:
    python run_inference_optimized.py --model plus --image "path/to/image.png"
    python run_inference_optimized.py --model L --image "path/to/image.png"
"""

import os, sys, time, argparse
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoConfig, GenerationConfig
import onnxruntime as ort

N = 8  # decoder layers

def kv_names(pref, n=N):
    r = []
    for i in range(n):
        for s in ['sk', 'sv', 'ck', 'cv']:
            r.append(f'{pref}_{i}_{s}')
    return r

def kv_names_self(pref, n=N):
    r = []
    for i in range(n):
        for s in ['sk', 'sv']:
            r.append(f'{pref}_{i}_{s}')
    return r

def decoder_shape_config(cfg):
    tc = cfg.text_config
    n_layers = int(getattr(tc, 'decoder_layers', None) or getattr(tc, 'num_hidden_layers'))
    n_heads = int(getattr(tc, 'decoder_attention_heads', None) or getattr(tc, 'num_attention_heads'))
    hidden = int(getattr(tc, 'd_model', None) or getattr(tc, 'hidden_size'))
    head_dim = int(getattr(tc, 'head_dim', None) or (hidden // n_heads))
    return n_layers, n_heads, head_dim

def generation_max_length(gen_cfg):
    mx = getattr(gen_cfg, 'max_length', None)
    if mx is None:
        raise ValueError("model generation max_length is not configured")
    return int(mx)

def empty_self_kv(batch, n_layers, n_heads, head_dim):
    return [
        np.zeros((batch, n_heads, 0, head_dim), dtype=np.float32)
        for _ in range(n_layers * 2)
    ]

def extract_split_self_kv(outputs, n_layers):
    pkv_all = list(outputs[1:])
    pkv_self = []
    for layer in range(n_layers):
        pkv_self.append(pkv_all[layer * 4])
        pkv_self.append(pkv_all[layer * 4 + 1])
    return pkv_self

def main():
    p = argparse.ArgumentParser(description="PP-FormulaNet KV-Cache Inference")
    p.add_argument('--model', default='plus', choices=['plus', 'L'])
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--max-steps', type=int, default=None,
                   help='Deprecated; ignored. Inference always uses model generation_config.max_length.')
    p.add_argument('--encoder', default=None, help='Path to encoder ONNX')
    p.add_argument('--merged', default=None, help='Path to merged decoder ONNX')
    p.add_argument('--prefill', default=None, help='Path to prefill decoder ONNX')
    p.add_argument('--decode', default=None, help='Path to decode decoder ONNX')
    p.add_argument('--processor', default=None, help='Path to model dir')
    args = p.parse_args()

    sfx = 'plus-L' if args.model == 'plus' else 'L'
    mname = f'PP-FormulaNet_{sfx}'
    mdir = args.processor or f'D:\\models\\{mname}_safetensors'
    ep = args.encoder or f'onnx/{mname}_encoder.onnx'
    mp = args.merged or f'onnx/{mname}_decoder_model_merged.onnx'

    use_split = bool(args.prefill or args.decode)
    if use_split and not (args.prefill and args.decode):
        print('ERROR: --prefill and --decode must be provided together for split decoder mode')
        sys.exit(1)
    pp = args.prefill
    dp = args.decode

    required = [args.image, ep, pp, dp] if use_split else [args.image, ep, mp]
    for f in required:
        if not os.path.exists(f):
            print(f'ERROR: not found: {f}')
            if not use_split and f == mp:
                print('       Export merged first, or pass both --prefill and --decode to use split models.')
            sys.exit(1)

    print('='*60)
    print(f'PP-FormulaNet KV-Cache Inference: {mname}')
    print('='*60)

    # Load processor + config
    print(f'\nLoading: {mdir}')
    proc = AutoProcessor.from_pretrained(mdir)
    cfg = AutoConfig.from_pretrained(mdir)
    gen_cfg = GenerationConfig.from_pretrained(mdir)
    sid = gen_cfg.decoder_start_token_id or cfg.text_config.eos_token_id
    eid = gen_cfg.eos_token_id or cfg.text_config.eos_token_id
    max_steps = generation_max_length(gen_cfg)
    n_layers, n_heads, head_dim = decoder_shape_config(cfg)
    print(f'  start={sid} eos={eid}')
    print(f'  max_length={max_steps}')
    if args.max_steps is not None and args.max_steps != max_steps:
        print(f'  NOTE: --max-steps={args.max_steps} ignored; inference uses model max_length={max_steps}')
    print(f'  decoder layers={n_layers} heads={n_heads} head_dim={head_dim}')

    # Process image
    print(f'\nImage: {args.image}')
    img = Image.open(args.image).convert('RGB')
    pv = proc(images=img, return_tensors='pt')['pixel_values'].numpy().astype(np.float32)
    print(f'  shape: {pv.shape}')

    enc = ort.InferenceSession(ep, providers=['CPUExecutionProvider'])
    if use_split:
        print(f'\nLoad ONNX: {ep}, {pp}, {dp}')
        pre = ort.InferenceSession(pp, providers=['CPUExecutionProvider'])
        dec = ort.InferenceSession(dp, providers=['CPUExecutionProvider'])
    else:
        print(f'\nLoad ONNX: {ep}, {mp}')
        dec = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])

    # Encoder
    print('\nEncoder...')
    t0 = time.time()
    enc_out = enc.run(None, {'pv': pv})[0].astype(np.float32)
    print(f'  {enc_out.shape} ({time.time()-t0:.2f}s)')

    kn_self = kv_names_self('past', n_layers)
    gen = [sid]

    # Decode loop
    print(f'\nGenerating (max_length {max_steps})...')
    st = time.time()
    if use_split:
        print('  mode: split prefill/decode')
        po = pre.run(None, {'ids': np.array([[sid]], dtype=np.int64), 'enc': enc_out})
        t = int(po[0].flatten()[0])
        gen.append(t)
        pkv_self = extract_split_self_kv(po, n_layers)
        decoder_calls = 1
        start_remaining = max_steps - 2
    else:
        print('  mode: merged decoder')
        t = sid
        pkv_self = empty_self_kv(pv.shape[0], n_layers, n_heads, head_dim)
        decoder_calls = 0
        start_remaining = max_steps - 1

    first_merged_call = not use_split
    for _ in range(start_remaining):
        if not first_merged_call and t == eid: break
        fd = {'ids': np.array([[t]], dtype=np.int64), 'enc': enc_out}
        for i, n in enumerate(kn_self): fd[n] = pkv_self[i]
        do = dec.run(None, fd)
        t = int(do[0].flatten()[0])
        gen.append(t); decoder_calls += 1
        if use_split:
            pkv_self = extract_split_self_kv(do, n_layers)
        else:
            pkv_self = list(do[1:])
        first_merged_call = False
    total = time.time() - st

    # Decode
    formula = proc.tokenizer.decode(gen, skip_special_tokens=True)

    # Report
    print(f'\n{"="*60}\nRESULTS\n{"="*60}')
    print(f'  Decoder mode: {"split" if use_split else "merged"}')
    print(f'  Tokens: {len(gen)}')
    print(f'  Decoder calls: {decoder_calls}')
    print(f'  Time: {total:.2f}s ({total/decoder_calls*1000:.1f}ms/call)' if decoder_calls > 0 else f'  Time: {total:.2f}s')
    print(f'\n  Formula: {formula}')

    # Transfer stats (16 self-KV tensors per step)
    kv_tensors = n_layers * 2
    current_kb = sum(x.nbytes for x in pkv_self) / 1024 if pkv_self else 0
    print(f'\n  KV tensors: {kv_tensors} self-KV tensors')
    print(f'  Current KV cache: ~{current_kb:.0f}KB')
    print(f'  Token transfer/step: 8B (vs 200KB for full logits)')
    print(f'  Total token output: {decoder_calls*8}B')
    return formula

if __name__ == '__main__':
    main()
