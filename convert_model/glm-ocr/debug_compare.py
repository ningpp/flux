"""
Deep comparison: Transformers vs ONNX token-by-token.

Runs both models step by step, comparing logits at each decode step
to find the root cause of token divergence.
"""
import os, sys, time, warnings
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
import onnxruntime as ort

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
MODEL_PATH  = r"D:\models\GLM-OCR"
ONNX_DIR    = r"D:\models\onnx\GLM-OCR"
IMAGE_PATH  = r"D:\tmp\formula-2026028-105537.jpg"
PROMPT      = "Formula Recognition:"
MAX_TOKENS  = 60  # enough to cover divergence at step 45
NUM_LAYERS  = 16
NUM_KV_HEADS = 8
HEAD_DIM    = 128
SPATIAL_MERGE_SIZE = 2

# ============================================================
# Helpers
# ============================================================
def compute_pos_ids(grid_thw, spatial_merge_size=SPATIAL_MERGE_SIZE):
    pos_ids_list = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // spatial_merge_size, spatial_merge_size,
                                     w // spatial_merge_size, spatial_merge_size)
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // spatial_merge_size, spatial_merge_size,
                                     w // spatial_merge_size, spatial_merge_size)
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        pos_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids_list, dim=0)
    max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_thw)
    return pos_ids, max_grid_size


def top_k_tokens(logits_1d, tokenizer, k=10):
    """Return top-k token ids, probs from a 1-D logits array."""
    if isinstance(logits_1d, torch.Tensor):
        logits_1d = logits_1d.detach().cpu().numpy()
    top_idx = np.argsort(logits_1d)[::-1][:k]
    # softmax
    max_l = logits_1d.max()
    exp_l = np.exp(logits_1d - max_l)
    probs = exp_l / exp_l.sum()
    rows = []
    for idx in top_idx:
        rows.append((int(idx), float(probs[idx]), float(logits_1d[idx]),
                      tokenizer.decode([int(idx)])))
    return rows

# ============================================================
# Main
# ============================================================
def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image: {image.size}")

    # ----- Load processor / tokenizer -----
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    image_token_id = config.image_token_id
    eos_token_id = getattr(config, "eos_token_id", None) or tokenizer.eos_token_id
    print(f"image_token_id={image_token_id}, eos_token_id={eos_token_id}")

    # ----- Prepare inputs (shared) -----
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]          # [1, S]
    pixel_values = inputs["pixel_values"]    # [N, C]
    image_grid_thw = inputs["image_grid_thw"]
    position_ids = inputs.get("position_ids")
    if position_ids is None:
        S = input_ids.shape[1]
        position_ids = torch.arange(S).unsqueeze(0).unsqueeze(0).expand(3, 1, S)

    seq_len = input_ids.shape[1]
    print(f"seq_len={seq_len}")

    # ================================================================
    # A) Transformers step-by-step with logit capture
    # ================================================================
    print("\n===== Loading Transformers model =====")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="float32")
    model.eval()

    # Run full generate and capture logits with a hook
    # Instead, we'll do manual step-by-step generation using model.forward
    
    # 1. Prefill: run model with full input
    with torch.no_grad():
        # Build inputs for model.forward
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "position_ids": position_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": True,
        }
        out = model(**model_inputs)
    
    tf_prefill_logits = out.logits[:, -1, :].detach().cpu().numpy()  # [1, V]
    past_kv = out.past_key_values
    tf_token0 = int(np.argmax(tf_prefill_logits[0]))
    print(f"[TF] Prefill -> token0 = {tf_token0} '{tokenizer.decode([tf_token0])}'")
    
    # Step-by-step decode
    tf_tokens = [tf_token0]
    tf_logits_per_step = [tf_prefill_logits[0].copy()]
    
    for step in range(1, MAX_TOKENS):
        next_id = torch.tensor([[tf_tokens[-1]]], dtype=torch.long)
        # position: all 3 dims advance together
        next_pos = position_ids[:, :, -1:] + step
        total = seq_len + step
        attn_mask = torch.ones((1, total), dtype=torch.long)
        
        with torch.no_grad():
            out = model(
                input_ids=next_id,
                position_ids=next_pos,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = out.past_key_values
        logits_np = out.logits[:, -1, :].detach().cpu().numpy()[0]
        tf_logits_per_step.append(logits_np.copy())
        
        tok = int(np.argmax(logits_np))
        tf_tokens.append(tok)
        if tok == eos_token_id or tok == 59246 or tok == 59253:
            break
    
    print(f"[TF] Generated {len(tf_tokens)} tokens")
    del model, past_kv, out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # ================================================================
    # B) ONNX step-by-step with logit capture
    # ================================================================
    print("\n===== Loading ONNX models =====")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ["CPUExecutionProvider"]
    
    vision_sess = ort.InferenceSession(os.path.join(ONNX_DIR, "vision_encoder.onnx"), opts, providers=prov)
    embed_sess  = ort.InferenceSession(os.path.join(ONNX_DIR, "embedding.onnx"), opts, providers=prov)
    prefill_sess = ort.InferenceSession(os.path.join(ONNX_DIR, "llm_prefill.onnx"), opts, providers=prov)
    decode_sess  = ort.InferenceSession(os.path.join(ONNX_DIR, "llm_unified.onnx"), opts, providers=prov)
    print("  All ONNX models loaded")

    # Numpy versions
    input_ids_np = input_ids.numpy()
    pv_np = pixel_values.numpy().astype(np.float32)
    igt_np = np.asarray(image_grid_thw).astype(np.int64)
    pos_ids_np = position_ids.numpy().astype(np.int64)
    attn_np = np.ones_like(input_ids_np, dtype=np.int64)

    # Vision encoder
    pos_ids_vision, max_grid = compute_pos_ids(igt_np)
    vis_out = vision_sess.run(None, {
        "pixel_values": pv_np,
        "pos_ids": pos_ids_vision.numpy().astype(np.int64),
        "max_grid_size": np.array(max_grid, dtype=np.int64),
    })[0]
    print(f"  Vision tokens: {vis_out.shape}")

    # Embedding
    tok_emb = embed_sess.run(None, {"input_ids": input_ids_np.astype(np.int64)})[0]

    # Merge
    inputs_embeds = tok_emb.copy()
    mask = input_ids_np == image_token_id
    vis_flat = vis_out.reshape(-1, vis_out.shape[-1])
    inputs_embeds[mask] = vis_flat

    # Prefill
    prefill_outputs = prefill_sess.run(None, {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attn_np,
        "position_ids": pos_ids_np,
    })
    ox_prefill_logits = prefill_outputs[0][:, -1, :]  # [1, V]
    kv_cache = []
    for i in range(NUM_LAYERS):
        kv_cache.append(prefill_outputs[1 + i * 2])
        kv_cache.append(prefill_outputs[2 + i * 2])
    
    ox_token0 = int(np.argmax(ox_prefill_logits[0]))
    print(f"[OX] Prefill -> token0 = {ox_token0} '{tokenizer.decode([ox_token0])}'")

    # Step-by-step decode
    ox_tokens = [ox_token0]
    ox_logits_per_step = [ox_prefill_logits[0].copy()]
    current_pos = pos_ids_np[:, :, -1:] + 1

    for step in range(1, MAX_TOKENS):
        next_id_np = np.array([[ox_tokens[-1]]], dtype=np.int64)
        next_emb = embed_sess.run(None, {"input_ids": next_id_np})[0]
        next_pos = current_pos + (step - 1)
        total = seq_len + step
        dec_attn = np.ones((1, total), dtype=np.int64)

        dec_inputs = {
            "inputs_embeds": next_emb.astype(np.float32),
            "attention_mask": dec_attn,
            "position_ids": next_pos.astype(np.int64),
        }
        for i in range(NUM_LAYERS):
            dec_inputs[f"past_key_{i}"] = kv_cache[i * 2].astype(np.float32)
            dec_inputs[f"past_value_{i}"] = kv_cache[i * 2 + 1].astype(np.float32)
        
        outputs = decode_sess.run(None, dec_inputs)
        logits = outputs[0][:, -1, :]  # [1, V]
        ox_logits_per_step.append(logits[0].copy())
        
        kv_cache = []
        for i in range(NUM_LAYERS):
            kv_cache.append(outputs[1 + i * 2])
            kv_cache.append(outputs[2 + i * 2])
        
        tok = int(np.argmax(logits[0]))
        ox_tokens.append(tok)
        if tok == eos_token_id or tok == 59246 or tok == 59253:
            break

    print(f"[OX] Generated {len(ox_tokens)} tokens")

    # ================================================================
    # C) Compare step by step
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 80)
    
    n = min(len(tf_tokens), len(ox_tokens), len(tf_logits_per_step), len(ox_logits_per_step))
    
    print(f"\n{'Step':>4} | {'TF tok':>7} {'TF text':>12} | {'OX tok':>7} {'OX text':>12} | {'Match':>5} | {'Logit MaxDiff':>13} | {'Logit CosSim':>12} | {'TF rank of OX':>13} | {'OX rank of TF':>13}")
    print("-" * 140)
    
    first_diff_step = None
    for step in range(n):
        tf_tok = tf_tokens[step]
        ox_tok = ox_tokens[step]
        match = "OK" if tf_tok == ox_tok else "DIFF"
        
        tf_l = tf_logits_per_step[step]
        ox_l = ox_logits_per_step[step]
        
        # Max absolute difference
        max_diff = float(np.max(np.abs(tf_l - ox_l)))
        
        # Cosine similarity
        dot = np.dot(tf_l, ox_l)
        norm_tf = np.linalg.norm(tf_l)
        norm_ox = np.linalg.norm(ox_l)
        cos_sim = dot / (norm_tf * norm_ox + 1e-12)
        
        # What rank does the OTHER model's chosen token have?
        tf_sorted = np.argsort(tf_l)[::-1]
        ox_sorted = np.argsort(ox_l)[::-1]
        tf_rank_of_ox = int(np.where(tf_sorted == ox_tok)[0][0]) + 1
        ox_rank_of_tf = int(np.where(ox_sorted == tf_tok)[0][0]) + 1
        
        tf_text = repr(tokenizer.decode([tf_tok]))
        ox_text = repr(tokenizer.decode([ox_tok]))
        
        print(f"{step:>4} | {tf_tok:>7} {tf_text:>12} | {ox_tok:>7} {ox_text:>12} | {match:>5} | {max_diff:>13.4f} | {cos_sim:>12.8f} | {tf_rank_of_ox:>13} | {ox_rank_of_tf:>13}")
        
        if match == "DIFF" and first_diff_step is None:
            first_diff_step = step
    
    # ================================================================
    # D) Detailed analysis at first divergence
    # ================================================================
    if first_diff_step is not None:
        s = first_diff_step
        print(f"\n{'=' * 80}")
        print(f"DETAILED ANALYSIS AT FIRST DIVERGENCE (step {s})")
        print(f"{'=' * 80}")
        
        print(f"\nTransformers top-10:")
        for rank, (tid, prob, logit, txt) in enumerate(top_k_tokens(tf_logits_per_step[s], tokenizer)):
            marker = " <-- OX chose" if tid == ox_tokens[s] else ""
            print(f"  #{rank+1:>2}: id={tid:>6}  prob={prob:.6f}  logit={logit:>10.4f}  '{txt}'{marker}")
        
        print(f"\nONNX top-10:")
        for rank, (tid, prob, logit, txt) in enumerate(top_k_tokens(ox_logits_per_step[s], tokenizer)):
            marker = " <-- TF chose" if tid == tf_tokens[s] else ""
            print(f"  #{rank+1:>2}: id={tid:>6}  prob={prob:.6f}  logit={logit:>10.4f}  '{txt}'{marker}")
        
        # Logit diff stats
        diff = tf_logits_per_step[s] - ox_logits_per_step[s]
        print(f"\nLogit difference stats:")
        print(f"  mean:   {np.mean(diff):.6f}")
        print(f"  std:    {np.std(diff):.6f}")
        print(f"  max:    {np.max(diff):.6f}")
        print(f"  min:    {np.min(diff):.6f}")
        print(f"  median: {np.median(diff):.6f}")
        print(f"  |diff| > 1.0 count: {np.sum(np.abs(diff) > 1.0)}")
        print(f"  |diff| > 0.5 count: {np.sum(np.abs(diff) > 0.5)}")
        print(f"  |diff| > 0.1 count: {np.sum(np.abs(diff) > 0.1)}")
        
        # Check logit for the two contested tokens
        tf_chosen = tf_tokens[s]
        ox_chosen = ox_tokens[s]
        print(f"\n  TF logit for TF-chosen ({tf_chosen}): {tf_logits_per_step[s][tf_chosen]:.6f}")
        print(f"  TF logit for OX-chosen ({ox_chosen}): {tf_logits_per_step[s][ox_chosen]:.6f}")
        print(f"  TF gap (TF - OX):                   {tf_logits_per_step[s][tf_chosen] - tf_logits_per_step[s][ox_chosen]:.6f}")
        print(f"  OX logit for TF-chosen ({tf_chosen}): {ox_logits_per_step[s][tf_chosen]:.6f}")
        print(f"  OX logit for OX-chosen ({ox_chosen}): {ox_logits_per_step[s][ox_chosen]:.6f}")
        print(f"  OX gap (OX - TF):                   {ox_logits_per_step[s][ox_chosen] - ox_logits_per_step[s][tf_chosen]:.6f}")

        # Also check the step BEFORE divergence
        if s > 0:
            prev = s - 1
            diff_prev = tf_logits_per_step[prev] - ox_logits_per_step[prev]
            print(f"\n  Step {prev} (before divergence) logit diff stats:")
            print(f"    max |diff|: {np.max(np.abs(diff_prev)):.6f}")
            print(f"    cos sim:    {np.dot(tf_logits_per_step[prev], ox_logits_per_step[prev]) / (np.linalg.norm(tf_logits_per_step[prev]) * np.linalg.norm(ox_logits_per_step[prev])):.8f}")
    else:
        print("\nNo divergence found in the first {n} steps!")

    # ================================================================
    # E) Track drift over all matching steps
    # ================================================================
    print(f"\n{'=' * 80}")
    print("DRIFT ACCUMULATION")
    print(f"{'=' * 80}")
    end = first_diff_step if first_diff_step else n
    for s in range(end):
        diff = tf_logits_per_step[s] - ox_logits_per_step[s]
        max_abs = np.max(np.abs(diff))
        cos = np.dot(tf_logits_per_step[s], ox_logits_per_step[s]) / (np.linalg.norm(tf_logits_per_step[s]) * np.linalg.norm(ox_logits_per_step[s]) + 1e-12)
        print(f"  step {s:>3}: max|diff|={max_abs:>10.4f}  cos_sim={cos:.8f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
