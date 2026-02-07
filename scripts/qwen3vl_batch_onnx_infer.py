"""
Qwen3-VL-2B-Instruct ONNX **Batch** Inference with KV-cache.
Run: D:\\conda\\envs\\qwen3vlonnx\\python scripts\\qwen3vl_batch_onnx_infer.py

Processes multiple images in a single batch through the decoder.
Each image is run through the vision encoder individually, then
all sequences are batched for the autoregressive decoder.

Since constrained_resize guarantees exactly EXPORT_PATCHES patches
(grid_h * grid_w = 288) for every image, the merged token count is
always 72, making all sequences the same length — no padding needed.
"""
import json
import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

MODEL_DIR = r"D:\models\Qwen3-VL-2B-Instruct"
ONNX_DIR = r"D:\models\onnx\Qwen3-VL-2B-Instruct"

# Read model config for dynamic constants
with open(os.path.join(ONNX_DIR, "config.json"), "r", encoding="utf-8") as f:
    _config = json.load(f)
_deepstack_indexes = _config.get("vision_config", {}).get("deepstack_visual_indexes", [8, 16, 24])
NUM_DEEPSTACK = len(_deepstack_indexes)

# Model constants (from config)
NUM_LAYERS = _config.get("text_config", {}).get("num_hidden_layers", 28)
NUM_KV_HEADS = _config.get("text_config", {}).get("num_key_value_heads", 8)
HEAD_DIM = _config.get("text_config", {}).get("head_dim", 128)
HIDDEN_SIZE = _config.get("text_config", {}).get("hidden_size", 2048)
VOCAB_SIZE = _config.get("text_config", {}).get("vocab_size", 151936)
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32

# Special token IDs
IM_START = 151644
IM_END = 151645
ENDOFTEXT = 151643
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
EOS_TOKENS = {IM_END, ENDOFTEXT}

# Preprocessing
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Constrained resize parameters — must match ONNX export
EXPORT_PATCHES = 288
MIN_GRID_DIM = 4
MAX_GRID_DIM = 24


def _build_valid_grids(target_patches, merge_size, min_dim, max_dim):
    grids = []
    for gh in range(min_dim, min(target_patches, max_dim) + 1, merge_size):
        if target_patches % gh == 0:
            gw = target_patches // gh
            if min_dim <= gw <= max_dim and gw % merge_size == 0:
                grids.append((gh, gw))
    return grids


_VALID_GRIDS = _build_valid_grids(EXPORT_PATCHES, MERGE_SIZE, MIN_GRID_DIM, MAX_GRID_DIM)
_VALID_SIZES = [(gh * PATCH_SIZE, gw * PATCH_SIZE) for gh, gw in _VALID_GRIDS]


def constrained_resize(height, width):
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {height}x{width}")
    aspect = width / height
    best = min(_VALID_SIZES, key=lambda hw: abs((hw[1] / hw[0]) - aspect))
    return best


def preprocess_image(image):
    """Preprocess a PIL Image -> pixel_values [num_patches, 1536], image_grid_thw [1, 3]."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    orig_w, orig_h = image.size
    new_h, new_w = constrained_resize(orig_h, orig_w)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    pixels = pixels.transpose(2, 0, 1)  # [3, H, W]
    C, H, W = pixels.shape

    pixels_tchw = np.stack([pixels, pixels], axis=0)  # [2, 3, H, W]

    grid_t = 2 // TEMPORAL_PATCH_SIZE
    grid_h = H // PATCH_SIZE
    grid_w = W // PATCH_SIZE

    patches = pixels_tchw.reshape(
        grid_t, TEMPORAL_PATCH_SIZE, C,
        grid_h // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        grid_w // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    num_patches = grid_t * grid_h * grid_w
    patch_dim = C * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    pixel_values = patches.reshape(num_patches, patch_dim).astype(np.float32)

    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, image_grid_thw


def build_input_ids(tokenizer, num_image_tokens):
    system_text = tokenizer.encode("system\nYou are a helpful assistant.").ids
    user_prefix = tokenizer.encode("user\n").ids
    user_text = tokenizer.encode("Convert this formula image to LaTeX.\n/no_think").ids
    assistant_text = tokenizer.encode("assistant\n").ids
    newline = tokenizer.encode("\n").ids

    input_ids = []
    input_ids.append(IM_START)
    input_ids.extend(system_text)
    input_ids.append(IM_END)
    input_ids.extend(newline)
    input_ids.append(IM_START)
    input_ids.extend(user_prefix)
    input_ids.append(VISION_START)
    input_ids.extend([IMAGE_PAD] * num_image_tokens)
    input_ids.append(VISION_END)
    input_ids.extend(user_text)
    input_ids.append(IM_END)
    input_ids.extend(newline)
    input_ids.append(IM_START)
    input_ids.extend(assistant_text)
    return np.array(input_ids, dtype=np.int64)


def compute_position_ids(input_ids, image_grid_thw):
    seq_len = len(input_ids)
    llm_pos_ids_list = []
    st = 0
    image_idx = 0
    vis_starts = [i for i, t in enumerate(input_ids) if t == VISION_START]

    for vis_start in vis_starts:
        img_start = vis_start + 1
        img_end = img_start
        while img_end < seq_len and input_ids[img_end] == IMAGE_PAD:
            img_end += 1

        t, h, w = image_grid_thw[image_idx]
        image_idx += 1
        llm_grid_t = t
        llm_grid_h = h // MERGE_SIZE
        llm_grid_w = w // MERGE_SIZE

        text_len = img_start - st
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        if text_len > 0:
            text_pos = np.arange(text_len, dtype=np.int64).reshape(1, -1)
            text_pos = np.broadcast_to(text_pos, (3, text_len)).copy() + st_idx
            llm_pos_ids_list.append(text_pos)

        t_index = np.arange(llm_grid_t).reshape(-1, 1)
        t_index = np.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w)).flatten()
        h_index = np.arange(llm_grid_h).reshape(1, -1, 1)
        h_index = np.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
        w_index = np.arange(llm_grid_w).reshape(1, 1, -1)
        w_index = np.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()

        vis_pos = np.stack([t_index, h_index, w_index])
        vis_pos = vis_pos + text_len + st_idx
        llm_pos_ids_list.append(vis_pos)
        st = img_end

    if st < seq_len:
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = seq_len - st
        text_pos = np.arange(text_len, dtype=np.int64).reshape(1, -1)
        text_pos = np.broadcast_to(text_pos, (3, text_len)).copy() + st_idx
        llm_pos_ids_list.append(text_pos)

    position_ids = np.concatenate(llm_pos_ids_list, axis=1)
    return position_ids.reshape(3, 1, seq_len).astype(np.int64)


def scatter_deepstack(features, input_ids, seq_len):
    scattered = np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)
    vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
    n = min(len(vis_positions), features.shape[0])
    for i in range(n):
        scattered[0, vis_positions[i], :] = features[i, :]
    return scattered


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def batch_infer(images, tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=50):
    """Run batch inference on a list of PIL Images.

    Args:
        images: list of PIL.Image.Image (RGB)
        tokenizer: tokenizers.Tokenizer
        vision_sess, embed_sess, decoder_sess: ORT sessions
        max_new_tokens: max tokens to generate per image

    Returns:
        list of str — generated text for each image
    """
    batch_size = len(images)
    print(f"\n{'='*60}")
    print(f"  Batch inference: {batch_size} images")
    print(f"{'='*60}")

    # --- 1. Preprocess & vision encode each image individually ---
    all_image_features = []   # list of [num_merged, hidden]
    all_deepstack = []        # list of [NUM_DEEPSTACK][num_merged, hidden]
    all_grid_thw = []         # list of [1, 3]

    t0 = time.time()
    for idx, img in enumerate(images):
        pixel_values, image_grid_thw = preprocess_image(img)
        all_grid_thw.append(image_grid_thw)

        vision_out = vision_sess.run(None, {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        })
        all_image_features.append(vision_out[0])
        all_deepstack.append([vision_out[1 + i] for i in range(NUM_DEEPSTACK)])

        gh, gw = image_grid_thw[0, 1], image_grid_thw[0, 2]
        print(f"  Image {idx}: {img.size[0]}x{img.size[1]} -> grid [{gh},{gw}], "
              f"merged={vision_out[0].shape[0]}")
    print(f"  Vision encoding: {time.time() - t0:.2f}s")

    # --- 2. Build input_ids for each image (all same length) ---
    # merged tokens = grid_t * (grid_h/merge) * (grid_w/merge) = 1 * (gh*gw)/(merge^2) = 288/4 = 72
    num_merged_tokens = all_image_features[0].shape[0]
    assert all(f.shape[0] == num_merged_tokens for f in all_image_features), \
        "All images must produce the same number of merged tokens"

    per_image_ids = [build_input_ids(tokenizer, num_merged_tokens) for _ in range(batch_size)]
    seq_len = len(per_image_ids[0])
    print(f"  seq_len={seq_len}, merged_tokens={num_merged_tokens}")

    # --- 3. Batch embed tokens ---
    batched_ids = np.stack(per_image_ids, axis=0)  # [batch, seq_len]
    inputs_embeds = embed_sess.run(None, {"input_ids": batched_ids})[0]  # [batch, seq_len, hidden]

    # --- 4. Replace IMAGE_PAD positions with vision features per image ---
    vis_positions = [i for i, tid in enumerate(per_image_ids[0]) if tid == IMAGE_PAD]
    for b in range(batch_size):
        for i, pos in enumerate(vis_positions):
            inputs_embeds[b, pos, :] = all_image_features[b][i, :]

    # --- 5. Compute position_ids per image, then stack ---
    # position_ids: [3, 1, seq_len] per image -> [3, batch, seq_len]
    per_image_pos = [compute_position_ids(per_image_ids[b], all_grid_thw[b][0:1])
                     for b in range(batch_size)]
    # Each is [3, 1, seq_len] — concatenate along batch dim
    position_ids = np.concatenate(per_image_pos, axis=1)  # [3, batch, seq_len]

    # --- 6. Scatter deepstack per image, then stack ---
    # Per image: scatter_deepstack returns [1, seq_len, hidden]
    # Stack to [batch, seq_len, hidden] for each deepstack level
    ds_scattered = []
    for di in range(NUM_DEEPSTACK):
        per_image_ds = [scatter_deepstack(all_deepstack[b][di], per_image_ids[b], seq_len)
                        for b in range(batch_size)]
        ds_scattered.append(np.concatenate(per_image_ds, axis=0))  # [batch, seq_len, hidden]

    # --- 7. Autoregressive decoding (batched) ---
    print("\n  --- Decoding ---")
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    past_kv = {}
    for i in range(NUM_LAYERS):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

    generated_ids = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    t_gen = time.time()

    for step in range(max_new_tokens):
        feeds = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for di in range(NUM_DEEPSTACK):
            feeds[f"deepstack_scattered_{di}"] = ds_scattered[di].astype(np.float32)
        feeds.update(past_kv)

        t_step = time.time()
        outputs = decoder_sess.run(None, feeds)
        logits = outputs[0]  # [batch, seq, vocab]

        # Greedy: pick argmax of last token for each batch
        next_tokens = []
        for b in range(batch_size):
            if finished[b]:
                # Use a dummy token (pad or EOS) for finished sequences
                next_tokens.append(ENDOFTEXT)
            else:
                tid = int(np.argmax(logits[b, -1, :]))
                generated_ids[b].append(tid)
                if tid in EOS_TOKENS:
                    finished[b] = True
                next_tokens.append(tid)

        if step == 0:
            print(f"  Prefill: {time.time() - t_step:.2f}s, first tokens: {next_tokens}")
        elif step < 5 or all(finished):
            print(f"  Step {step+1}: tokens={next_tokens}, time={time.time() - t_step:.3f}s")

        if all(finished):
            print(f"  All sequences finished at step {step + 1}")
            break

        # Update KV cache
        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past_kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]

        # Next token embeddings: [batch, 1]
        next_ids = np.array([[t] for t in next_tokens], dtype=np.int64)  # [batch, 1]
        inputs_embeds = embed_sess.run(None, {"input_ids": next_ids})[0]  # [batch, 1, hidden]

        # Position_ids: max+1 for all 3 dims (text tokens)
        max_pos = int(position_ids.max()) + 1
        position_ids = np.full((3, batch_size, 1), max_pos, dtype=np.int64)

        # Attention mask grows
        total_len = attention_mask.shape[1] + 1
        attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

        # Deepstack: zeros for decode steps
        ds_scattered = [np.zeros((batch_size, 1, HIDDEN_SIZE), dtype=np.float32)
                        for _ in range(NUM_DEEPSTACK)]

    gen_time = time.time() - t_gen
    print(f"\n  Generated in {gen_time:.2f}s")

    # Decode tokens to text
    results = []
    for b in range(batch_size):
        text = tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        results.append(text)
        print(f"  Image {b}: [{len(generated_ids[b])} tokens] {text[:80]}...")

    return results, generated_ids


def single_infer(image, tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=50):
    """Run single-image inference (for validation comparison)."""
    pixel_values, image_grid_thw = preprocess_image(image)
    vision_out = vision_sess.run(None, {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    })
    image_features = vision_out[0]
    deepstack_feats = [vision_out[1 + i] for i in range(NUM_DEEPSTACK)]

    num_vis_tokens = image_features.shape[0]
    input_ids = build_input_ids(tokenizer, num_vis_tokens)
    seq_len = len(input_ids)

    position_ids = compute_position_ids(input_ids, image_grid_thw[0:1])

    input_ids_2d = input_ids.reshape(1, -1)
    inputs_embeds = embed_sess.run(None, {"input_ids": input_ids_2d})[0]

    vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
    for i, pos in enumerate(vis_positions):
        inputs_embeds[0, pos, :] = image_features[i, :]

    ds_scattered = [scatter_deepstack(deepstack_feats[i], input_ids, seq_len)
                    for i in range(NUM_DEEPSTACK)]

    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    past_kv = {}
    for i in range(NUM_LAYERS):
        past_kv[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        past_kv[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

    generated_ids = []
    for step in range(max_new_tokens):
        feeds = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for di in range(NUM_DEEPSTACK):
            feeds[f"deepstack_scattered_{di}"] = ds_scattered[di].astype(np.float32)
        feeds.update(past_kv)

        outputs = decoder_sess.run(None, feeds)
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token_id)

        if next_token_id in EOS_TOKENS:
            break

        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past_kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]

        next_ids = np.array([[next_token_id]], dtype=np.int64)
        inputs_embeds = embed_sess.run(None, {"input_ids": next_ids})[0]
        max_pos = int(position_ids.max()) + 1
        position_ids = np.full((3, 1, 1), max_pos, dtype=np.int64)
        total_len = attention_mask.shape[1] + 1
        attention_mask = np.ones((1, total_len), dtype=np.int64)
        ds_scattered = [np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32) for _ in range(NUM_DEEPSTACK)]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


def main():
    print("=" * 60)
    print("Qwen3-VL-2B  ONNX Batch Inference Test")
    print("=" * 60)

    tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("Loading ONNX models...")
    t0 = time.time()
    vision_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "vision_encoder.onnx"), opts, providers=["CPUExecutionProvider"])
    embed_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "embed_tokens.onnx"), opts, providers=["CPUExecutionProvider"])
    decoder_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "decoder_model_merged.onnx"), opts, providers=["CPUExecutionProvider"])
    print(f"Models loaded in {time.time() - t0:.2f}s")

    # --- Test 1: two small random images ---
    print("\n" + "=" * 60)
    print("TEST 1: Two small random images (batch=2)")
    print("=" * 60)
    np.random.seed(42)
    img1 = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8), "RGB")
    img2 = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8), "RGB")

    batch_texts, batch_ids = batch_infer(
        [img1, img2], tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=20)

    # --- Test 2: verify batch == sequential ---
    print("\n" + "=" * 60)
    print("TEST 2: Verify batch results match sequential")
    print("=" * 60)
    seq_text1, seq_ids1 = single_infer(img1, tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=20)
    seq_text2, seq_ids2 = single_infer(img2, tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=20)

    match1 = batch_ids[0] == seq_ids1
    match2 = batch_ids[1] == seq_ids2
    print(f"\n  Image 0 - batch vs seq match: {match1}")
    print(f"    Batch:      {batch_ids[0][:15]}...")
    print(f"    Sequential: {seq_ids1[:15]}...")
    print(f"  Image 1 - batch vs seq match: {match2}")
    print(f"    Batch:      {batch_ids[1][:15]}...")
    print(f"    Sequential: {seq_ids2[:15]}...")

    if match1 and match2:
        print("\n  BATCH == SEQUENTIAL: PERFECT MATCH!")
    else:
        print("\n  WARNING: batch and sequential results differ!")
        print("  This is expected if the decoder's attention mechanism")
        print("  behaves differently with padding in the KV-cache.")

    # --- Test 3: batch of 3 with same image repeated ---
    print("\n" + "=" * 60)
    print("TEST 3: Batch of 3 (same image repeated)")
    print("=" * 60)
    batch3_texts, batch3_ids = batch_infer(
        [img1, img1, img1], tokenizer, vision_sess, embed_sess, decoder_sess, max_new_tokens=10)

    all_same = batch3_ids[0] == batch3_ids[1] == batch3_ids[2]
    print(f"\n  All 3 outputs identical: {all_same}")
    if not all_same:
        for b in range(3):
            print(f"    Batch {b}: {batch3_ids[b]}")

    print("\n" + "=" * 60)
    print("ALL BATCH TESTS DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
