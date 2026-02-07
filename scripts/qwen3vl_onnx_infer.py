"""
Qwen3-VL-2B-Instruct ONNX Inference with KV-cache.
Run: D:\\conda\\envs\\qwen3vlonnx\\python scripts\\qwen3vl_onnx_infer.py

Uses 3 ONNX models:
  1. vision_encoder.onnx    - image features + deepstack
  2. embed_tokens.onnx      - token embeddings
  3. decoder_model_merged.onnx - autoregressive decoder with KV-cache

Key design: Deepstack uses PRE-SCATTERED tensors [batch, seq, hidden].
The caller scatters vision features into the right positions before calling decoder.
During decode steps, deepstack tensors are all zeros.
Deepstack count and model constants are read from config.json.
"""
import json
import os
import math
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

MODEL_DIR = r"D:\models\Qwen3-VL-2B-Instruct"
ONNX_DIR = r"D:\models\onnx\Qwen3-VL-2B-Instruct"
IMAGE_PATH = r"D:\tmp\formula_2025-8-2_17-28-16.jpg"

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
MIN_PIXELS = 65536
MAX_PIXELS = 16777216


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """Resize dimensions to multiples of factor, respecting pixel count limits."""
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {height}x{width}, min factor={factor}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(max_pixels / (height * width))
        h_bar = math.floor(height * beta / factor) * factor
        w_bar = math.floor(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(image_path: str):
    """Load and preprocess image -> pixel_values [num_patches, 1536], image_grid_thw [1, 3].

    Matches transformers Qwen2VLImageProcessorFast._preprocess:
    patches are grouped by merge_size (2x2) before flattening.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    new_h, new_w = smart_resize(orig_h, orig_w)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    print(f"  Image: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    # Normalize: [0,255] -> [0,1] -> (x - 0.5) / 0.5
    pixels = np.array(img, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD  # [H, W, 3]
    pixels = pixels.transpose(2, 0, 1)  # [3, H, W]
    C, H, W = pixels.shape

    # Duplicate for temporal_patch_size=2: [T, C, H, W]
    pixels_tchw = np.stack([pixels, pixels], axis=0)  # [2, 3, H, W]

    grid_t = 2 // TEMPORAL_PATCH_SIZE
    grid_h = H // PATCH_SIZE
    grid_w = W // PATCH_SIZE

    # Reshape matching transformers: group by merge_size
    # [grid_t, temp, C, gh/merge, merge_h, patch_h, gw/merge, merge_w, patch_w]
    patches = pixels_tchw.reshape(
        grid_t, TEMPORAL_PATCH_SIZE, C,
        grid_h // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        grid_w // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
    )
    # Permute to [grid_t, gh/merge, gw/merge, merge_h, merge_w, C, temp, patch_h, patch_w]
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    num_patches = grid_t * grid_h * grid_w
    patch_dim = C * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1536
    pixel_values = patches.reshape(num_patches, patch_dim).astype(np.float32)

    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    print(f"  pixel_values: {pixel_values.shape}, grid_thw: {image_grid_thw.tolist()}")
    return pixel_values, image_grid_thw


def build_input_ids(tokenizer, num_image_tokens: int):
    """Build prompt input_ids matching the torch reference template."""
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
    """Compute MRoPE 3D position_ids [3, 1, seq_len].

    - Text tokens: all 3 dims share same incrementing position
    - Image tokens: T=constant, H=row, W=col in merged grid
    """
    seq_len = len(input_ids)

    # Find image_pad regions
    llm_pos_ids_list = []
    st = 0
    image_idx = 0

    # Find all vision_start positions
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

        # Text tokens before image (including vision_start)
        text_len = img_start - st

        # Compute st_idx ONCE (matching transformers get_rope_index)
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        if text_len > 0:
            text_pos = np.arange(text_len, dtype=np.int64).reshape(1, -1)
            text_pos = np.broadcast_to(text_pos, (3, text_len)).copy() + st_idx
            llm_pos_ids_list.append(text_pos)

        # Image tokens: T/H/W grid (uses same st_idx as text)
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

    # Remaining text
    if st < seq_len:
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = seq_len - st
        text_pos = np.arange(text_len, dtype=np.int64).reshape(1, -1)
        text_pos = np.broadcast_to(text_pos, (3, text_len)).copy() + st_idx
        llm_pos_ids_list.append(text_pos)

    position_ids = np.concatenate(llm_pos_ids_list, axis=1)
    return position_ids.reshape(3, 1, seq_len).astype(np.int64)


def scatter_deepstack(features, input_ids, seq_len):
    """Scatter [num_vis, hidden] -> [1, seq_len, hidden] at IMAGE_PAD positions."""
    scattered = np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)
    vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
    n = min(len(vis_positions), features.shape[0])
    for i in range(n):
        scattered[0, vis_positions[i], :] = features[i, :]
    return scattered


def main():
    print("=" * 60)
    print("Qwen3-VL-2B-Instruct  -  ONNX Inference (KV-cache)")
    print("=" * 60)

    tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))

    t0 = time.time()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("Loading ONNX models...")
    vision_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "vision_encoder.onnx"), opts, providers=["CPUExecutionProvider"])
    embed_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "embed_tokens.onnx"), opts, providers=["CPUExecutionProvider"])
    decoder_sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "decoder_model_merged.onnx"), opts, providers=["CPUExecutionProvider"])
    print(f"Models loaded in {time.time() - t0:.2f}s")

    # 1. Preprocess image
    print("\n--- Preprocessing ---")
    pixel_values, image_grid_thw = preprocess_image(IMAGE_PATH)
    num_vis_tokens = image_grid_thw[0, 0] * (image_grid_thw[0, 1] // MERGE_SIZE) * (image_grid_thw[0, 2] // MERGE_SIZE)
    print(f"  Merged vision tokens: {num_vis_tokens}")

    # 2. Vision encoder
    print("\n--- Vision Encoder ---")
    t1 = time.time()
    vision_out = vision_sess.run(None, {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    })
    image_features = vision_out[0]
    deepstack_features = [vision_out[1 + i] for i in range(NUM_DEEPSTACK)]
    print(f"  image_features: {image_features.shape}, deepstack_count: {NUM_DEEPSTACK}, time: {time.time() - t1:.2f}s")

    # 3. Build input_ids
    print("\n--- Building Input IDs ---")
    input_ids = build_input_ids(tokenizer, num_vis_tokens)
    seq_len = len(input_ids)
    print(f"  input_ids length: {seq_len}")
    print(f"  input_ids: {input_ids.tolist()}")

    # 4. Position IDs
    position_ids = compute_position_ids(input_ids, image_grid_thw[0:1])
    print(f"  position_ids shape: {position_ids.shape}")

    # 5. Embed tokens
    print("\n--- Embedding ---")
    input_ids_2d = input_ids.reshape(1, -1)
    inputs_embeds = embed_sess.run(None, {"input_ids": input_ids_2d})[0]

    # 6. Replace image_pad embeddings with vision features
    vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
    for i, pos in enumerate(vis_positions):
        inputs_embeds[0, pos, :] = image_features[i, :]
    print(f"  Replaced {len(vis_positions)} image_pad embeddings")

    # 7. Scatter deepstack
    ds_scattered = [scatter_deepstack(deepstack_features[i], input_ids, seq_len) for i in range(NUM_DEEPSTACK)]

    # 8. Autoregressive decoding
    print("\n--- Decoding ---")
    max_new_tokens = 50
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    past_kv = {}
    for i in range(NUM_LAYERS):
        past_kv[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        past_kv[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

    generated_ids = []
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
        logits = outputs[0]

        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token_id)

        if step == 0:
            print(f"  Prefill: {time.time() - t_step:.2f}s, first token: {next_token_id}")
        else:
            print(f"  Step {step+1}: token={next_token_id}, time={time.time() - t_step:.3f}s")

        if next_token_id in EOS_TOKENS:
            print(f"  EOS at step {step + 1}")
            break

        # Update KV cache
        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past_kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]

        # Next token embedding
        next_ids = np.array([[next_token_id]], dtype=np.int64)
        inputs_embeds = embed_sess.run(None, {"input_ids": next_ids})[0]

        # Position_ids: max + 1, all 3 dims same for text tokens
        max_pos = int(position_ids.max()) + 1
        position_ids = np.full((3, 1, 1), max_pos, dtype=np.int64)

        # Attention mask grows
        total_len = attention_mask.shape[1] + 1
        attention_mask = np.ones((1, total_len), dtype=np.int64)

        # Deepstack: zeros for decode steps
        ds_scattered = [np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32) for _ in range(NUM_DEEPSTACK)]

    gen_time = time.time() - t_gen
    print(f"\n  Generated {len(generated_ids)} tokens in {gen_time:.2f}s")
    print(f"  Token IDs: {generated_ids}")

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n--- GENERATED TEXT ---")
    print(text)
    print(f"--- END ---")

    # Validation
    ground_truth = [73594, 64680, 198, 77, 0, 1124, 48053, 1124, 26888, 90, 17, 59, 2493, 308, 92, 1124, 2359, 7, 1124, 37018, 91362, 15170, 68, 92, 1124, 1291, 29776, 77, 198, 73594, 151645]
    print(f"\n--- VALIDATION ---")
    print(f"  Ground truth IDs: {ground_truth[:15]}...")
    print(f"  ONNX output IDs:  {generated_ids[:15]}...")
    n = min(len(ground_truth), len(generated_ids))
    match_count = sum(1 for i in range(n) if ground_truth[i] == generated_ids[i])
    print(f"  Match: {match_count}/{n} tokens")
    if match_count == n and n == len(ground_truth):
        print("  PERFECT MATCH!")


if __name__ == "__main__":
    main()
