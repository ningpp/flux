"""Quick debug inference for one image."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoTokenizer

# MODEL_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0")
MODEL_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0-FP16")
IMAGE_PATH = Path(r"D:\tmp\formula_2025-8-2_17-28-16.jpg")

IMAGE_SIZE = 1024
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = 64
MAX_LENGTH = 50
STOP_TOKEN_ID = 151645
IMAGE_TOKEN_ID = 151859
IMG_PAD_COUNT = 256


def preprocess_image(image_path: Path) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...].astype(np.float16)


def build_prompt(tokenizer):
    imgpad = "<imgpad>" * IMG_PAD_COUNT
    return (
        "<|im_start|>system\n"
        "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
        f"<img>{imgpad}</img>\n"
        " OCR with format: <|im_end|><|im_start|>assistant\n"
    )


def main():
    print(f"Model dir: {MODEL_DIR}")
    print(f"Image: {IMAGE_PATH}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CPUExecutionProvider"]

    vision = ort.InferenceSession(str(MODEL_DIR / "vision_encoder.onnx"), sess_options, providers=providers)
    embed = ort.InferenceSession(str(MODEL_DIR / "embed_tokens.onnx"), sess_options, providers=providers)
    decoder = ort.InferenceSession(str(MODEL_DIR / "decoder_model.onnx"), sess_options, providers=providers)
    print("Sessions loaded")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    prompt_ids = tokenizer.encode(build_prompt(tokenizer), add_special_tokens=False)
    prompt_ids = np.array([prompt_ids], dtype=np.int64)
    print(f"Prompt length: {prompt_ids.shape[1]}")

    pixel_values = preprocess_image(IMAGE_PATH)
    print(f"pixel_values: {pixel_values.shape}")

    image_features = vision.run(None, {"pixel_values": pixel_values})[0]
    print(f"image_features: {image_features.shape}")

    inputs_embeds = embed.run(None, {"input_ids": prompt_ids})[0]
    print(f"inputs_embeds: {inputs_embeds.shape}")

    # Replace imgpad positions
    merged = inputs_embeds.copy()
    feat_idx = 0
    for pos in range(prompt_ids.shape[1]):
        if prompt_ids[0, pos] == IMAGE_TOKEN_ID:
            merged[0, pos] = image_features[0, feat_idx]
            feat_idx += 1
    print(f"Replaced {feat_idx} imgpad tokens")

    batch_size = 1
    seq_len = merged.shape[1]
    total_len = seq_len

    feed = {
        "inputs_embeds": merged,
        "attention_mask": np.ones((batch_size, total_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(NUM_LAYERS):
        feed[f"past_key_{i}"] = np.empty((batch_size, NUM_HEADS, 0, HEAD_DIM), dtype=np.float16)
        feed[f"past_value_{i}"] = np.empty((batch_size, NUM_HEADS, 0, HEAD_DIM), dtype=np.float16)

    print("Running prefill...")
    outputs = decoder.run(None, feed)
    logits = outputs[0]
    present_kv = outputs[1:]
    print(f"logits: {logits.shape}")

    first = int(np.argmax(logits[0, -1, :]))
    print(f"first token: {first} -> {repr(tokenizer.decode([first]))}")

    generated = [first]
    for step in range(MAX_LENGTH):
        total_len += 1
        next_id = np.array([[generated[-1]]], dtype=np.int64)
        next_embeds = embed.run(None, {"input_ids": next_id})[0]
        feed = {
            "inputs_embeds": next_embeds,
            "attention_mask": np.ones((batch_size, total_len), dtype=np.int64),
            "position_ids": np.full((batch_size, 1), total_len - 1, dtype=np.int64),
        }
        for i in range(NUM_LAYERS):
            feed[f"past_key_{i}"] = present_kv[2 * i]
            feed[f"past_value_{i}"] = present_kv[2 * i + 1]

        outputs = decoder.run(None, feed)
        logits = outputs[0]
        present_kv = outputs[1:]
        next_token = int(np.argmax(logits[0, 0, :]))
        generated.append(next_token)
        print(f"step {step}: {next_token} -> {repr(tokenizer.decode([next_token]))}")
        if next_token == STOP_TOKEN_ID:
            break

    print("\nFinal:", tokenizer.decode(generated))


if __name__ == "__main__":
    main()
