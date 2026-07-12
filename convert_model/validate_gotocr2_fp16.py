"""
Validate the mixed-FP16 GOT-OCR-2.0 ONNX model against the FP32 baseline.

Steps:
1. Run FP32 ONNX on all formula images in D:\tmp and save reference token sequences.
2. Run FP16 ONNX on the same images.
3. Compare per-image token accuracy and full-match rate.
4. Report overall accuracy drop; fail if >= 1%.

Run with train_demo conda environment:
    conda run -n train_demo python convert_model\validate_gotocr2_fp16.py
"""
import os
import re
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

FP32_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0")
FP16_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0-FP16")
IMAGE_DIR = Path(r"D:\tmp")
OUTPUT_DIR = Path(r"D:\code\flux\output")
REFERENCE_PATH = OUTPUT_DIR / "fp32_reference_gotocr2.pkl"
FP16_RESULTS_PATH = OUTPUT_DIR / "fp16_results_gotocr2.pkl"
REPORT_PATH = OUTPUT_DIR / "gotocr2_fp16_validation_report.json"

IMAGE_SIZE = 1024
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float16)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float16)
RESCALE_FACTOR = np.float16(1.0 / 255.0)

NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = 64
DEFAULT_MAX_LENGTH = 1024
STOP_TOKEN_ID = 151645  # <|im_end|>, matches Java GotOcr2DecoderModel
IMAGE_TOKEN_ID = 151859
IMG_PAD_COUNT = 256


def build_prompt(tokenizer) -> str:
    imgpad = "<imgpad>" * IMG_PAD_COUNT
    return (
        "<|im_start|>system\n"
        "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
        f"<img>{imgpad}</img>\n"
        " OCR with format: <|im_end|><|im_start|>assistant\n"
    )


def preprocess_image(image_path: Path, float_dtype: np.dtype = np.float32) -> np.ndarray:
    """Resize to 1024x1024, rescale, normalize, transpose to CHW."""
    mean = IMAGE_MEAN.astype(float_dtype)
    std = IMAGE_STD.astype(float_dtype)
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=float_dtype) * float_dtype(RESCALE_FACTOR)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...]  # [1, 3, 1024, 1024]


def get_float_dtype(model_dir: Path) -> np.dtype:
    """Infer whether the model expects FP32 or FP16 I/O from vision encoder input."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(str(model_dir / "vision_encoder.onnx"), sess_options, providers=["CPUExecutionProvider"])
    pixel_type = session.get_inputs()[0].type
    return np.float16 if pixel_type == "tensor(float16)" else np.float32


def create_session(model_dir: Path, use_gpu: bool = False) -> Tuple[ort.InferenceSession, ort.InferenceSession, ort.InferenceSession]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    # ORT_ENABLE_ALL can fuse LayerNorm incorrectly with the FP16 decoder graph,
    # so use EXTENDED which keeps the graph valid while still applying most optimizations.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    vision = ort.InferenceSession(str(model_dir / "vision_encoder.onnx"), sess_options, providers=providers)
    embed = ort.InferenceSession(str(model_dir / "embed_tokens.onnx"), sess_options, providers=providers)
    decoder = ort.InferenceSession(str(model_dir / "decoder_model.onnx"), sess_options, providers=providers)
    return vision, embed, decoder


def run_encoder(vision_session: ort.InferenceSession, pixel_values: np.ndarray) -> np.ndarray:
    return vision_session.run(None, {"pixel_values": pixel_values})[0]


def run_embed(embed_session: ort.InferenceSession, input_ids: np.ndarray) -> np.ndarray:
    return embed_session.run(None, {"input_ids": input_ids})[0]


def prepare_inputs_embeds(input_ids: np.ndarray, image_features: np.ndarray, inputs_embeds: np.ndarray) -> np.ndarray:
    """Replace <imgpad> positions in inputs_embeds with image_features."""
    result = inputs_embeds.copy()
    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        feat_idx = 0
        for pos in range(input_ids.shape[1]):
            if input_ids[b, pos] == IMAGE_TOKEN_ID:
                result[b, pos] = image_features[b, feat_idx]
                feat_idx += 1
    return result


def decode_autoregressive(
    decoder_session: ort.InferenceSession,
    embed_session: ort.InferenceSession,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    float_dtype: np.dtype = np.float32,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> List[int]:
    """Greedy autoregressive decoding matching Java GotOcr2DecoderModel."""
    batch_size = inputs_embeds.shape[0]
    seq_len = inputs_embeds.shape[1]
    total_len = seq_len

    # Initial prefill inputs
    attention_mask = np.ones((batch_size, total_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)

    feed = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(NUM_LAYERS):
        feed[f"past_key_{i}"] = np.empty((batch_size, NUM_HEADS, 0, HEAD_DIM), dtype=float_dtype)
        feed[f"past_value_{i}"] = np.empty((batch_size, NUM_HEADS, 0, HEAD_DIM), dtype=float_dtype)

    outputs = decoder_session.run(None, feed)
    logits = outputs[0]
    present_kv = outputs[1:]

    # First generated token from last prefill position
    first_token = int(np.argmax(logits[0, -1, :]))
    generated = [first_token]
    if first_token == STOP_TOKEN_ID:
        return generated

    for step in range(max_length):
        total_len += 1
        next_id = np.array([[generated[-1]]], dtype=np.int64)
        next_embeds = run_embed(embed_session, next_id)

        attention_mask = np.ones((batch_size, total_len), dtype=np.int64)
        position_ids = np.full((batch_size, 1), total_len - 1, dtype=np.int64)

        feed = {
            "inputs_embeds": next_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(NUM_LAYERS):
            feed[f"past_key_{i}"] = present_kv[2 * i]
            feed[f"past_value_{i}"] = present_kv[2 * i + 1]

        outputs = decoder_session.run(None, feed)
        logits = outputs[0]
        present_kv = outputs[1:]

        next_token = int(np.argmax(logits[0, 0, :]))
        generated.append(next_token)
        if next_token == STOP_TOKEN_ID:
            break

    return generated


def run_inference(model_dir: Path, image_paths: List[Path], use_gpu: bool = False, max_length: int = DEFAULT_MAX_LENGTH) -> Dict[str, List[int]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = build_prompt(tokenizer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_ids = np.array([prompt_ids], dtype=np.int64)

    float_dtype = get_float_dtype(model_dir)
    vision, embed, decoder = create_session(model_dir, use_gpu=use_gpu)
    results: Dict[str, List[int]] = {}

    for idx, img_path in enumerate(image_paths):
        print(f"  [{idx + 1}/{len(image_paths)}] {img_path.name}")
        pixel_values = preprocess_image(img_path, float_dtype)
        image_features = run_encoder(vision, pixel_values)
        inputs_embeds = run_embed(embed, prompt_ids)
        merged_embeds = prepare_inputs_embeds(prompt_ids, image_features, inputs_embeds)
        generated_tokens = decode_autoregressive(decoder, embed, prompt_ids, merged_embeds, float_dtype, max_length=max_length)
        results[img_path.name] = generated_tokens

    return results


def token_accuracy(ref: List[int], hyp: List[int]) -> float:
    """Token-level accuracy, aligned by shorter length."""
    if not ref:
        return 1.0 if not hyp else 0.0
    length = min(len(ref), len(hyp))
    if length == 0:
        return 0.0
    matches = sum(1 for a, b in zip(ref[:length], hyp[:length]) if a == b)
    return matches / max(len(ref), len(hyp))


def levenshtein_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def normalize_text(text: str) -> str:
    """Strip whitespace for semantic comparison of LaTeX output."""
    return "".join(ch for ch in text if not ch.isspace())


def is_formula_image(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return False
    # Heuristic: filename contains formula-related keyword
    keywords = ["formula", "math", "equa", "latex", "ocr-rec", "ccf0c84d", "dolphin",
                "bar-simple", "table", "layout-formula", "text-with-formula", "yu-zhou"]
    return any(k in name for k in keywords)


def collect_formula_images(directory: Path) -> List[Path]:
    images = [p for p in directory.iterdir() if is_formula_image(p)]
    # Sort deterministically
    images.sort(key=lambda p: p.name)
    print(f"Found {len(images)} formula images in {directory}")
    return images


def compute_metrics(ref_results: Dict[str, List[int]], fp16_results: Dict[str, List[int]], tokenizer) -> Dict:
    accuracies = []
    norm_accuracies = []
    full_matches = 0
    per_image = []
    common_names = [name for name in ref_results if name in fp16_results]
    for name in common_names:
        ref = ref_results[name]
        hyp = fp16_results[name]
        acc = token_accuracy(ref, hyp)
        full = ref == hyp
        accuracies.append(acc)
        full_matches += int(full)

        ref_text = tokenizer.decode(ref, skip_special_tokens=True)
        hyp_text = tokenizer.decode(hyp, skip_special_tokens=True)
        norm_ref = normalize_text(ref_text)
        norm_hyp = normalize_text(hyp_text)
        dist = levenshtein_distance(norm_ref, norm_hyp)
        denom = max(len(norm_ref), len(norm_hyp))
        norm_acc = 1.0 if denom == 0 else 1.0 - dist / denom
        norm_accuracies.append(norm_acc)

        per_image.append({
            "image": name,
            "token_accuracy": round(acc, 6),
            "normalized_text_accuracy": round(norm_acc, 6),
            "full_match": full,
            "ref_len": len(ref),
            "fp16_len": len(hyp),
        })

    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
    avg_norm_acc = float(np.mean(norm_accuracies)) if norm_accuracies else 0.0
    full_match_rate = full_matches / len(accuracies) if accuracies else 0.0
    return {
        "avg_token_accuracy": round(avg_acc, 6),
        "avg_normalized_text_accuracy": round(avg_norm_acc, 6),
        "full_match_rate": round(full_match_rate, 6),
        "images_compared": len(accuracies),
        "full_matches": full_matches,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate GOT-OCR-2.0 FP16 model against FP32 reference.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDAExecutionProvider for inference")
    parser.add_argument("--skip-fp32", action="store_true", help="Skip FP32 reference run if pickle exists")
    parser.add_argument("--only-fp32", action="store_true", help="Generate FP32 reference and exit")
    parser.add_argument("--strict-formula", action="store_true", help="Only use images whose filename contains 'formula'")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum generated tokens per image")
    parser.add_argument("--compare-only", action="store_true", help="Compare existing reference/FP16 pickles without inference")
    args = parser.parse_args()

    print("=" * 60)
    print("GOT-OCR-2.0 FP16 Validation")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        if not REFERENCE_PATH.exists() or not FP16_RESULTS_PATH.exists():
            raise FileNotFoundError(f"--compare-only requires {REFERENCE_PATH} and {FP16_RESULTS_PATH}")
        print(f"\nCompare-only mode: loading {REFERENCE_PATH} and {FP16_RESULTS_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(str(FP32_DIR), trust_remote_code=True)
        with open(REFERENCE_PATH, "rb") as f:
            ref_results = pickle.load(f)
        with open(FP16_RESULTS_PATH, "rb") as f:
            fp16_results = pickle.load(f)
    else:
        image_paths = collect_formula_images(IMAGE_DIR)
        if not image_paths:
            raise RuntimeError(f"No formula images found in {IMAGE_DIR}")
        if args.strict_formula:
            image_paths = [p for p in image_paths if "formula" in p.name.lower()]
            print(f"Strict formula filter: {len(image_paths)} images")
        if args.max_images > 0:
            image_paths = image_paths[:args.max_images]

        # 1. FP32 reference
        if REFERENCE_PATH.exists() and args.skip_fp32:
            print(f"\nLoading existing FP32 reference from {REFERENCE_PATH}")
            with open(REFERENCE_PATH, "rb") as f:
                ref_results = pickle.load(f)
        else:
            print("\nRunning FP32 reference inference...")
            ref_results = run_inference(FP32_DIR, image_paths, use_gpu=args.use_gpu, max_length=args.max_length)
            with open(REFERENCE_PATH, "wb") as f:
                pickle.dump(ref_results, f)
            print(f"Saved FP32 reference to {REFERENCE_PATH}")

            if args.only_fp32:
                print("\n--only-fp32 specified; exiting after reference generation.")
                return 0

        # 2. FP16 inference
        print("\nRunning FP16 inference...")
        fp16_results = run_inference(FP16_DIR, image_paths, use_gpu=args.use_gpu, max_length=args.max_length)
        with open(FP16_RESULTS_PATH, "wb") as f:
            pickle.dump(fp16_results, f)
        print(f"Saved FP16 results to {FP16_RESULTS_PATH}")

    # 3. Compare
    print("\nComparing FP16 against FP32 reference...")
    tokenizer = AutoTokenizer.from_pretrained(str(FP32_DIR), trust_remote_code=True)
    metrics = compute_metrics(ref_results, fp16_results, tokenizer)
    token_accuracy_drop = 1.0 - metrics["avg_token_accuracy"]
    norm_accuracy_drop = 1.0 - metrics["avg_normalized_text_accuracy"]
    metrics["token_accuracy_drop"] = round(token_accuracy_drop, 6)
    metrics["normalized_accuracy_drop"] = round(norm_accuracy_drop, 6)
    # Pass/fail is based on semantic (whitespace-normalized) text accuracy.
    metrics["passes_1pct_threshold"] = norm_accuracy_drop < 0.01
    metrics["max_length"] = args.max_length

    # 4. Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 5. Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Images compared: {metrics['images_compared']}")
    print(f"Average token accuracy: {metrics['avg_token_accuracy'] * 100:.4f}%")
    print(f"Token accuracy drop: {token_accuracy_drop * 100:.4f}%")
    print(f"Average normalized text accuracy: {metrics['avg_normalized_text_accuracy'] * 100:.4f}%")
    print(f"Normalized accuracy drop: {norm_accuracy_drop * 100:.4f}%")
    print(f"Full match rate: {metrics['full_match_rate'] * 100:.2f}%")
    print(f"Full matches: {metrics['full_matches']}/{metrics['images_compared']}")
    print(f"Passes <1% threshold (normalized): {metrics['passes_1pct_threshold']}")
    print(f"Report saved to: {REPORT_PATH}")

    if not metrics["passes_1pct_threshold"]:
        print("\nWARNING: FP16 model does NOT meet the <1% accuracy requirement.")
        return 1
    print("\nSUCCESS: FP16 model meets the <1% accuracy requirement.")
    return 0


if __name__ == "__main__":
    exit(main())
