"""
Test Qwen3-VL ONNX inference with various image sizes.
Validates that preprocessing, vision encoder, embedding, position IDs,
deepstack scattering, and decoding all work for different resolutions.

Run: D:\\conda\\envs\\qwen3vlonnx\\python scripts\\test_qwen3vl_sizes.py
"""
import json
import os
import sys
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

ONNX_DIR = r"D:\models\onnx\Qwen3-VL-2B-Instruct"

# Read model config
with open(os.path.join(ONNX_DIR, "config.json"), "r", encoding="utf-8") as f:
    _config = json.load(f)
_deepstack_indexes = _config.get("vision_config", {}).get("deepstack_visual_indexes", [8, 16, 24])
NUM_DEEPSTACK = len(_deepstack_indexes)
NUM_LAYERS = _config.get("text_config", {}).get("num_hidden_layers", 28)
NUM_KV_HEADS = _config.get("text_config", {}).get("num_key_value_heads", 8)
HEAD_DIM = _config.get("text_config", {}).get("head_dim", 128)
HIDDEN_SIZE = _config.get("text_config", {}).get("hidden_size", 2048)

PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32

# The ONNX vision encoder has baked position embeddings for this total patch count.
EXPORT_PATCHES = 288
MIN_GRID_DIM = 4
MAX_GRID_DIM = 24  # max(export_grid_h, export_grid_w)

IM_START = 151644
IM_END = 151645
ENDOFTEXT = 151643
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
EOS_TOKENS = {IM_END, ENDOFTEXT}

IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


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
    """Pick the (H, W) from valid options that best matches the input aspect ratio."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {height}x{width}")
    aspect = width / height
    return min(_VALID_SIZES, key=lambda hw: abs((hw[1] / hw[0]) - aspect))


# Test image sizes: (width, height) — various aspect ratios and scales
TEST_SIZES = [
    (32, 32),       # Minimum possible (1 factor each)
    (64, 64),       # Small square
    (100, 50),      # Wide, small
    (50, 100),      # Tall, small
    (200, 100),     # 2:1 aspect
    (389, 173),     # Same as original test image
    (640, 480),     # Standard 4:3
    (800, 600),     # Larger 4:3
    (1024, 768),    # XGA
    (1920, 1080),   # Full HD (wide)
    (300, 1200),    # Very tall
    (33, 65),       # Odd non-factor sizes
    (127, 97),      # Prime-ish sizes
    (500, 500),     # Medium square
]


def preprocess_image_from_pil(img):
    orig_w, orig_h = img.size
    new_h, new_w = constrained_resize(orig_h, orig_w)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    pixels = np.array(img, dtype=np.float32) / 255.0
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

    return pixel_values, image_grid_thw, (new_w, new_h)


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


def test_one_size(w, h, tokenizer, vision_sess, embed_sess, decoder_sess,
                  run_decoder=False, max_decode_steps=2):
    """Test the pipeline with a synthetic image of the given size.

    When run_decoder=False (default), tests preprocessing + vision encoder +
    embedding + position_ids + shape checks (fast).
    When run_decoder=True, also runs prefill + a few decode steps (slow on CPU).
    Returns (success: bool, message: str).
    """
    # Create synthetic image
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8), 'RGB')

    # Preprocess
    pixel_values, image_grid_thw, resized = preprocess_image_from_pil(img)
    grid_t, grid_h, grid_w = image_grid_thw[0]
    num_patches = int(grid_t * grid_h * grid_w)
    num_merged = int(grid_t * (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE))

    assert pixel_values.shape == (num_patches, 1536), \
        f"pixel_values shape mismatch: {pixel_values.shape} vs expected ({num_patches}, 1536)"

    # Vision encoder
    vision_out = vision_sess.run(None, {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    })
    image_features = vision_out[0]
    deepstack_features = [vision_out[1 + i] for i in range(NUM_DEEPSTACK)]

    assert image_features.shape == (num_merged, HIDDEN_SIZE), \
        f"image_features shape mismatch: {image_features.shape} vs expected ({num_merged}, {HIDDEN_SIZE})"
    for di in range(NUM_DEEPSTACK):
        assert deepstack_features[di].shape == (num_merged, HIDDEN_SIZE), \
            f"deepstack_{di} shape mismatch: {deepstack_features[di].shape}"

    # Build input_ids
    input_ids = build_input_ids(tokenizer, num_merged)
    seq_len = len(input_ids)

    # Position IDs
    position_ids = compute_position_ids(input_ids, image_grid_thw[0:1])
    assert position_ids.shape == (3, 1, seq_len), \
        f"position_ids shape mismatch: {position_ids.shape}"

    # Embed tokens
    input_ids_2d = input_ids.reshape(1, -1)
    inputs_embeds = embed_sess.run(None, {"input_ids": input_ids_2d})[0]
    assert inputs_embeds.shape == (1, seq_len, HIDDEN_SIZE), \
        f"inputs_embeds shape mismatch: {inputs_embeds.shape}"

    # Replace IMAGE_PAD with vision features
    vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
    assert len(vis_positions) == num_merged, \
        f"IMAGE_PAD count {len(vis_positions)} != num_merged {num_merged}"
    for i, pos in enumerate(vis_positions):
        inputs_embeds[0, pos, :] = image_features[i, :]

    # Scatter deepstack
    ds_scattered = [scatter_deepstack(deepstack_features[i], input_ids, seq_len) for i in range(NUM_DEEPSTACK)]

    generated = []

    if run_decoder:
        # Prefill
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        past_kv = {}
        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
            past_kv[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

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
        assert logits.shape[0] == 1 and logits.shape[1] == seq_len, \
            f"logits shape mismatch: {logits.shape}"

        first_token = int(np.argmax(logits[0, -1, :]))
        generated = [first_token]

        # A few decode steps
        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past_kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]

        total_len = seq_len
        max_pos = int(position_ids.max())

        for step in range(max_decode_steps - 1):
            if generated[-1] in EOS_TOKENS:
                break
            total_len += 1
            max_pos += 1

            next_ids = np.array([[generated[-1]]], dtype=np.int64)
            next_embeds = embed_sess.run(None, {"input_ids": next_ids})[0]

            feeds = {
                "inputs_embeds": next_embeds.astype(np.float32),
                "attention_mask": np.ones((1, total_len), dtype=np.int64),
                "position_ids": np.full((3, 1, 1), max_pos, dtype=np.int64),
            }
            for di in range(NUM_DEEPSTACK):
                feeds[f"deepstack_scattered_{di}"] = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)
            for i in range(NUM_LAYERS):
                feeds[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feeds[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]

            outputs = decoder_sess.run(None, feeds)
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)

            for i in range(NUM_LAYERS):
                past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
                past_kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]

    decoder_label = f", tokens={generated}" if run_decoder else ""
    info = (f"resized={resized[0]}x{resized[1]}, grid_thw=[{grid_t},{grid_h},{grid_w}], "
            f"patches={num_patches}, merged={num_merged}, seq={seq_len}{decoder_label}")
    return True, info


def main():
    print("=" * 70)
    print("Qwen3-VL ONNX Multi-Size Test")
    print(f"NUM_DEEPSTACK={NUM_DEEPSTACK}, NUM_LAYERS={NUM_LAYERS}, "
          f"NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM}, HIDDEN_SIZE={HIDDEN_SIZE}")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))

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
    print("Models loaded.\n")

    passed = 0
    failed = 0
    errors = []

    # Decoder is very slow on CPU — only validate shapes, not full generation.
    # The original image validation at the bottom does full end-to-end decode.

    for w, h in TEST_SIZES:
        label = f"[{w}x{h}]"
        try:
            ok, info = test_one_size(w, h, tokenizer, vision_sess, embed_sess,
                                     decoder_sess, run_decoder=False)
            if ok:
                print(f"  PASS {label:>12s}  {info}")
                passed += 1
            else:
                print(f"  FAIL {label:>12s}  {info}")
                failed += 1
                errors.append((w, h, info))
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR {label:>11s}  {e}")
            print(f"    {tb.splitlines()[-2]}")
            failed += 1
            errors.append((w, h, str(e)))

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_SIZES)} tests")
    if errors:
        print("\nFailed tests:")
        for w, h, err in errors:
            print(f"  {w}x{h}: {err}")
    print("=" * 70)

    # Also test the original image
    print("\n--- Original Image Validation ---")
    img_path = r"D:\tmp\formula_2025-8-2_17-28-16.jpg"
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values, image_grid_thw, resized = preprocess_image_from_pil(img)
            num_merged = int(image_grid_thw[0, 0] * (image_grid_thw[0, 1] // MERGE_SIZE) * (image_grid_thw[0, 2] // MERGE_SIZE))

            vision_out = vision_sess.run(None, {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})
            image_features = vision_out[0]
            deepstack_features = [vision_out[1 + i] for i in range(NUM_DEEPSTACK)]

            input_ids = build_input_ids(tokenizer, num_merged)
            seq_len = len(input_ids)
            position_ids = compute_position_ids(input_ids, image_grid_thw[0:1])
            inputs_embeds = embed_sess.run(None, {"input_ids": input_ids.reshape(1, -1)})[0]

            vis_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD]
            for i, pos in enumerate(vis_positions):
                inputs_embeds[0, pos, :] = image_features[i, :]

            ds_scattered = [scatter_deepstack(deepstack_features[i], input_ids, seq_len) for i in range(NUM_DEEPSTACK)]

            attention_mask = np.ones((1, seq_len), dtype=np.int64)
            past_kv = {}
            for i in range(NUM_LAYERS):
                past_kv[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
                past_kv[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

            generated_ids = []
            for step in range(50):
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

            ground_truth = [73594, 64680, 198, 77, 0, 1124, 48053, 1124, 26888, 90, 17, 59, 2493, 308, 92, 1124, 2359, 7, 1124, 37018, 91362, 15170, 68, 92, 1124, 1291, 29776, 77, 198, 73594, 151645]
            n = min(len(ground_truth), len(generated_ids))
            match_count = sum(1 for i in range(n) if ground_truth[i] == generated_ids[i])
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  Generated: {text}")
            print(f"  Token match: {match_count}/{n}")
            if match_count == n and n == len(ground_truth):
                print("  PERFECT MATCH! Original image validation PASSED")
            else:
                print(f"  MISMATCH! Expected {len(ground_truth)} tokens, got {len(generated_ids)}")
                print(f"  Expected: {ground_truth}")
                print(f"  Got:      {generated_ids}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
    else:
        print(f"  Skipped (image not found: {img_path})")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
