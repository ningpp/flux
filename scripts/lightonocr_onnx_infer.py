"""
LightOnOCR-2-1B ONNX Inference with KV-cache.
Run: D:\\conda\\envs\\qwen3vlonnx\\python scripts\\lightonocr_onnx_infer.py

Three ONNX sessions: vision_encoder, embed_tokens, decoder_model_merged.
Greedy decoding (argmax) for deterministic output.
"""
import math
import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from PIL import Image

MODEL_DIR = r"D:\models\LightOnOCR-2-1B-ONNX"
IMAGE_PATH = r"D:\tmp\formula_2025-8-2_17-28-16.jpg"

# --- Config ---
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
EFFECTIVE_PATCH_SIZE = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28
LONGEST_EDGE = 1540
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMAGE_TOKEN_ID = 151655       # <|image_pad|>
VISION_BREAK_TOKEN = "<|vision_pad|>"
VISION_END_TOKEN   = "<|vision_end|>"
IMAGE_PAD_TOKEN    = "<|image_pad|>"
EOS_TOKEN_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_NEW_TOKENS = 4096


def pixtral_resize(h: int, w: int) -> tuple[int, int]:
    """Resize so longest edge <= LONGEST_EDGE, dims are multiples of EFFECTIVE_PATCH_SIZE (28).
    The ONNX vision encoder's patch merger requires even patch grid dimensions.
    """
    ratio = max(h / LONGEST_EDGE, w / LONGEST_EDGE)
    if ratio > 1:
        h = int(math.floor(h / ratio))
        w = int(math.floor(w / ratio))
    # Round UP to nearest multiple of EFFECTIVE_PATCH_SIZE (28)
    new_h = ((h - 1) // EFFECTIVE_PATCH_SIZE + 1) * EFFECTIVE_PATCH_SIZE
    new_w = ((w - 1) // EFFECTIVE_PATCH_SIZE + 1) * EFFECTIVE_PATCH_SIZE
    return new_h, new_w


def preprocess_image(image_path: str) -> tuple[np.ndarray, int, int]:
    """Load + preprocess image → pixel_values [1, 3, H, W], num_rows, num_cols."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    new_h, new_w = pixtral_resize(orig_h, orig_w)
    print(f"Image: {orig_w}x{orig_h} → resize to {new_w}x{new_h}")

    img = img.resize((new_w, new_h), Image.BICUBIC)

    # To float32, rescale, normalize
    pixels = np.array(img, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD

    # HWC → CHW → NCHW
    pixels = pixels.transpose(2, 0, 1)[np.newaxis, ...]

    num_rows = new_h // EFFECTIVE_PATCH_SIZE
    num_cols = new_w // EFFECTIVE_PATCH_SIZE
    print(f"Token grid: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} image tokens")
    return pixels, num_rows, num_cols


def build_prompt(num_rows: int, num_cols: int) -> str:
    """Build full chat prompt with flat image tokens (no breaks/end tokens)."""
    num_image_tokens = num_rows * num_cols
    image_tokens = IMAGE_PAD_TOKEN * num_image_tokens
    prompt = (
        f"<|im_start|>system<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{image_tokens}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def main():
    print("=" * 60)
    print("LightOnOCR-2-1B  —  ONNX Inference with KV-cache (greedy)")
    print("=" * 60)

    # --- Load sessions ---
    t0 = time.time()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    encoder_sess = ort.InferenceSession(
        f"{MODEL_DIR}/vision_encoder.onnx", opts, providers=providers)
    embed_sess   = ort.InferenceSession(
        f"{MODEL_DIR}/embed_tokens.onnx", opts, providers=providers)
    decoder_sess = ort.InferenceSession(
        f"{MODEL_DIR}/decoder_model_merged.onnx", opts, providers=providers)
    tokenizer    = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    print(f"Sessions loaded in {time.time() - t0:.2f}s")

    # --- Preprocess image ---
    t1 = time.time()
    pixel_values, num_rows, num_cols = preprocess_image(IMAGE_PATH)

    # --- Build prompt & tokenize ---
    prompt = build_prompt(num_rows, num_cols)
    encoded = tokenizer.encode(prompt)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"input_ids[:20] = {input_ids[0, :20].tolist()}")
    num_image_tokens_in_prompt = np.sum(input_ids == IMAGE_TOKEN_ID)
    print(f"Number of <|image_pad|> tokens in prompt: {num_image_tokens_in_prompt}")
    prep_time = time.time() - t1
    print(f"Preprocessing: {prep_time:.3f}s")

    # --- Vision encoder ---
    t2 = time.time()
    image_features = encoder_sess.run(
        ["image_features"], {"pixel_values": pixel_values}
    )[0]  # [1, num_merged_patches, 1024]
    enc_time = time.time() - t2
    print(f"Vision encoder: {enc_time:.3f}s, image_features shape={image_features.shape}")
    assert image_features.shape[1] == num_rows * num_cols, \
        f"Mismatch: encoder output {image_features.shape[1]} patches vs grid {num_rows * num_cols}"

    # --- Token embedding ---
    t3 = time.time()
    inputs_embeds = embed_sess.run(
        ["inputs_embeds"], {"input_ids": input_ids}
    )[0]  # [1, seq_len, 1024]
    embed_time = time.time() - t3
    print(f"Token embedding: {embed_time:.3f}s, inputs_embeds shape={inputs_embeds.shape}")

    # --- Merge image features into text embeddings ---
    image_positions = np.where(input_ids[0] == IMAGE_TOKEN_ID)[0]
    for idx_in_features, pos in enumerate(image_positions):
        inputs_embeds[0, pos, :] = image_features[0, idx_in_features, :]
    print(f"Merged {len(image_positions)} image features into text embeddings")

    # --- Prefill ---
    t4 = time.time()
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    # Empty KV cache
    decoder_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }
    for i in range(NUM_LAYERS):
        decoder_inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        decoder_inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

    decoder_output_names = [o.name for o in decoder_sess.get_outputs()]
    prefill_result = decoder_sess.run(decoder_output_names, decoder_inputs)

    # Parse outputs
    output_map = dict(zip(decoder_output_names, prefill_result))
    logits = output_map["logits"]  # [1, seq_len, vocab]
    kv_cache = {}
    for i in range(NUM_LAYERS):
        kv_cache[f"past_key_values.{i}.key"] = output_map[f"present.{i}.key"]
        kv_cache[f"past_key_values.{i}.value"] = output_map[f"present.{i}.value"]

    # First token from prefill
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_tokens = [next_token]
    prefill_time = time.time() - t4
    print(f"Prefill: {prefill_time:.3f}s")

    # --- Decode loop ---
    t5 = time.time()
    curr_len = seq_len
    for step in range(MAX_NEW_TOKENS - 1):
        if next_token in EOS_TOKEN_IDS:
            break

        curr_len += 1
        # Embed next token
        next_input_ids = np.array([[next_token]], dtype=np.int64)
        next_embeds = embed_sess.run(
            ["inputs_embeds"], {"input_ids": next_input_ids}
        )[0]  # [1, 1, 1024]

        # Attention mask grows
        new_attention_mask = np.ones((1, curr_len), dtype=np.int64)

        # Build decoder inputs
        decoder_inputs = {
            "inputs_embeds": next_embeds,
            "attention_mask": new_attention_mask,
        }
        decoder_inputs.update(kv_cache)

        # Run decoder
        step_result = decoder_sess.run(decoder_output_names, decoder_inputs)
        output_map = dict(zip(decoder_output_names, step_result))

        logits = output_map["logits"]
        for i in range(NUM_LAYERS):
            kv_cache[f"past_key_values.{i}.key"] = output_map[f"present.{i}.key"]
            kv_cache[f"past_key_values.{i}.value"] = output_map[f"present.{i}.value"]

        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)

    decode_time = time.time() - t5
    total_time = time.time() - t1

    # --- Decode text ---
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\nDecode loop: {decode_time:.2f}s  ({len(generated_tokens)} tokens)")
    print(f"Total inference: {total_time:.2f}s")
    print(f"First 10 generated token IDs: {generated_tokens[:10]}")
    print(f"\n--- GENERATED TEXT ---")
    print(text)
    print(f"--- END ---")


if __name__ == "__main__":
    main()
