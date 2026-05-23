from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_falcon_ocr_to_onnx import (
    build_batch_model_inputs,
    build_model_inputs,
)


def patch_transformers_runtime_for_cpu() -> None:
    """Allow Falcon-OCR remote code to run in the transformers5 CPU torch env."""
    import torch.nn.attention.flex_attention as flex_attention

    class AuxRequest:
        def __init__(self, lse: bool = False):
            self.lse = lse

    flex_attention.AuxRequest = AuxRequest
    torch.compile = lambda fn, *args, **kwargs: fn

    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        triton.jit = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn
        triton.cdiv = lambda x, y: (x + y - 1) // y
        lang = types.ModuleType("triton.language")

        class _ConstExpr:
            pass

        lang.constexpr = _ConstExpr
        for name in ["arange", "program_id", "load", "store", "where"]:
            setattr(lang, name, lambda *args, **kwargs: None)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = lang


class SimpleBlockMask:
    def __init__(self, mask: torch.Tensor, mask_mod):
        self.mask = mask
        self.mask_mod = mask_mod
        self.BLOCK_SIZE = (128, 128)
        self.seq_lengths = (mask.shape[-2], mask.shape[-1])

    def __getitem__(self, item):
        del item
        clone = SimpleBlockMask(self.mask, self.mask_mod)
        clone.seq_lengths = self.seq_lengths
        return clone


class AuxOutput:
    pass


def patch_loaded_falcon_model(model) -> None:
    module = sys.modules[model.__class__.__module__]

    def squared_relu_gate(packed: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        del hidden_dim
        return torch.relu(packed[..., 0::2]).square() * packed[..., 1::2]

    def create_batch_attention_mask(
        input_batch: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_id: int,
        soi_token_id: int,
        eoi_token_id: int,
        max_len: int | None = None,
    ) -> SimpleBlockMask:
        batch, input_len = input_batch.size()
        seq = max_len or input_len
        padded = torch.full((batch, seq), pad_token_id, dtype=input_batch.dtype)
        padded[:, :input_len] = input_batch

        positions = torch.arange(seq)
        causal = positions.view(1, seq, 1) >= positions.view(1, 1, seq)

        eos_mask = padded == eos_token_id
        eos_mask[:, -1] = True
        cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
        sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int64)
        sequence_indices[:, 1:] = cumulative_mask[:, :-1]
        document = sequence_indices[:, :, None] == sequence_indices[:, None, :]

        non_pad_mask_id = torch.cumsum(padded != pad_token_id, dim=1)
        non_left_pad = non_pad_mask_id[:, None, :] > 0
        block_causal = causal & document & non_left_pad

        soi_mask = padded == soi_token_id
        eoi_mask = padded == eoi_token_id
        acc_soi_mask = torch.cumsum(soi_mask.to(torch.int64), dim=1)
        acc_eoi_mask = torch.cumsum(eoi_mask.to(torch.int64), dim=1)
        img_mask = (acc_soi_mask - acc_eoi_mask) > 0
        img_indices = acc_soi_mask * img_mask.to(torch.int64)
        image_prefix = (
            img_mask[:, :, None]
            & img_mask[:, None, :]
            & (img_indices[:, :, None] == img_indices[:, None, :])
        )
        mask = image_prefix | block_causal

        def mask_mod(b, h, q, kv):
            del h
            return mask[b, q, kv]

        return SimpleBlockMask(mask, mask_mod)

    def dense_flex_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: SimpleBlockMask,
        return_aux=None,
        **kwargs,
    ):
        del return_aux, kwargs
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        query_len = query.shape[-2]
        key_len = key.shape[-2]
        if query_len == 1:
            mask = block_mask.mask[:, key_len - 1 : key_len, :key_len]
        else:
            mask = block_mask.mask[:, :query_len, :key_len]
        scores = scores.masked_fill(~mask[:, None, :, :], -torch.finfo(scores.dtype).max)
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)
        aux = AuxOutput()
        aux.lse = torch.logsumexp(scores, dim=-1)
        return output, aux

    module.squared_relu_gate = squared_relu_gate
    module.create_batch_attention_mask = create_batch_attention_mask
    module.compiled_flex_attn_decode = dense_flex_attention
    module.compiled_flex_attn_prefill = dense_flex_attention


def infer_category(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("formula"):
        return "formula"
    if name.startswith("table"):
        return "table"
    return "text"


def decode_generated(tokenizer, ids: list[int]) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text.replace("<|end_of_query|>", "").replace("<|end_of_text|>", "").strip()


def load_transformers_model(model_dir: Path):
    patch_transformers_runtime_for_cpu()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=torch.float32,
        local_files_only=True,
    ).eval()
    patch_loaded_falcon_model(model)
    model._ensure_device_buffers()
    return model


def load_kv_cache_shape(model_dir: Path, batch_size: int) -> tuple[int, int, int, int, int, int]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return (
        int(config["n_layers"]),
        2,
        batch_size,
        int(config["n_heads"]),
        0,
        int(config["head_dim"]),
    )


def generate_with_transformers(
    model,
    image_path: Path,
    category: str,
    *,
    max_new_tokens: int,
    min_dimension: int,
    max_dimension: int,
) -> dict:
    module = sys.modules[model.__class__.__module__]
    tokenizer = model._get_tokenizer()
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    model._pad_token_id = pad_token_id
    stop_token_ids = [model.config.eos_id, tokenizer.convert_tokens_to_ids("<|end_of_query|>")]

    instruction = module.CATEGORY_PROMPTS.get(category, module.CATEGORY_PROMPTS["plain"])
    prompt = f"<|image|>{instruction}\n<|OCR_PLAIN|>"

    start = time.perf_counter()
    batch_inputs = module.process_batch(
        tokenizer,
        model.config,
        [(str(image_path), prompt)],
        max_length=4096,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_sec = time.perf_counter() - start

    tokens = batch_inputs["tokens"]
    batch, prompt_len = tokens.size()
    block_size = 128
    total_seq = (prompt_len + max_new_tokens + block_size - 1) // block_size * block_size
    total_seq = min(total_seq, model.config.max_seq_len)
    kv_cache = module.KVCache(
        max_batch_size=batch,
        max_seq_length=total_seq,
        n_heads=model.config.n_heads,
        head_dim=model.config.head_dim,
        num_layers=model.config.n_layers,
    )
    padded_tokens = torch.full((batch, total_seq), pad_token_id, dtype=tokens.dtype)
    padded_tokens[:, :prompt_len] = tokens
    attention_mask = model.get_attention_mask(padded_tokens, max_len=total_seq)

    infer_start = time.perf_counter()
    with torch.inference_mode():
        logits = model.forward(
            tokens=tokens,
            rope_pos_t=batch_inputs["pos_t"],
            rope_pos_hw=batch_inputs["pos_hw"],
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            pixel_values=batch_inputs["pixel_values"],
            pixel_mask=batch_inputs["pixel_mask"],
        )
        generated_ids: list[int] = []
        stop_reason = "max_new_tokens"
        while len(generated_ids) < max_new_tokens and kv_cache.get_pos() < total_seq:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(torch.long)
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)
            if token_id in stop_token_ids:
                stop_reason = "stop_token"
                break
            pos = kv_cache.get_pos()
            padded_tokens[:, pos] = next_token[:, -1]
            logits = model.forward(
                tokens=next_token,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
    inference_sec = time.perf_counter() - infer_start

    return {
        "ids": generated_ids,
        "text": decode_generated(tokenizer, generated_ids),
        "prompt_length": int(prompt_len),
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason,
        "preprocess_sec": preprocess_sec,
        "inference_sec": inference_sec,
        "total_sec": preprocess_sec + inference_sec,
    }


def make_ort_session_with_provider(onnx_path: Path, provider: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    if provider == "auto":
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    else:
        providers = [provider]
    return ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)


def ortvalue_from_numpy_for_session(
    session: ort.InferenceSession,
    value: np.ndarray,
) -> ort.OrtValue:
    device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    return ort.OrtValue.ortvalue_from_numpy(value, device)


def run_unified_kv_iobinding(
    session: ort.InferenceSession,
    tokens: np.ndarray,
    image_patches: np.ndarray,
    pos_t: np.ndarray,
    pos_hw: np.ndarray,
    attention_mask: np.ndarray,
    past_key_values: ort.OrtValue,
) -> tuple[np.ndarray, ort.OrtValue]:
    io = session.io_binding()
    io.bind_ortvalue_input("tokens", ortvalue_from_numpy_for_session(session, tokens))
    io.bind_ortvalue_input("image_patches", ortvalue_from_numpy_for_session(session, image_patches))
    io.bind_ortvalue_input("pos_t", ortvalue_from_numpy_for_session(session, pos_t))
    io.bind_ortvalue_input("pos_hw", ortvalue_from_numpy_for_session(session, pos_hw))
    io.bind_ortvalue_input("attention_mask", ortvalue_from_numpy_for_session(session, attention_mask))
    io.bind_ortvalue_input("past_key_values", past_key_values)
    io.bind_output("logits", "cpu")
    kv_device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    io.bind_output("present_key_values", kv_device)
    session.run_with_iobinding(io)
    outputs = io.get_outputs()
    logits = outputs[0].numpy()
    present_key_values = outputs[1]
    return logits, present_key_values


def run_unified_kv_token_iobinding(
    session: ort.InferenceSession,
    tokens: np.ndarray,
    image_patches: np.ndarray,
    pos_t: np.ndarray,
    pos_hw: np.ndarray,
    attention_mask: np.ndarray,
    past_key_values: ort.OrtValue,
) -> tuple[np.ndarray, ort.OrtValue]:
    io = session.io_binding()
    io.bind_ortvalue_input("tokens", ortvalue_from_numpy_for_session(session, tokens))
    io.bind_ortvalue_input("image_patches", ortvalue_from_numpy_for_session(session, image_patches))
    io.bind_ortvalue_input("pos_t", ortvalue_from_numpy_for_session(session, pos_t))
    io.bind_ortvalue_input("pos_hw", ortvalue_from_numpy_for_session(session, pos_hw))
    io.bind_ortvalue_input("attention_mask", ortvalue_from_numpy_for_session(session, attention_mask))
    io.bind_ortvalue_input("past_key_values", past_key_values)
    io.bind_output("next_token", "cpu")
    kv_device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    io.bind_output("present_key_values", kv_device)
    session.run_with_iobinding(io)
    outputs = io.get_outputs()
    next_token = outputs[0].numpy()
    present_key_values = outputs[1]
    return next_token, present_key_values


def session_outputs_token_id(session: ort.InferenceSession) -> bool:
    return any(output.name == "next_token" for output in session.get_outputs())


def run_next_token_iobinding(
    session: ort.InferenceSession,
    tokens: np.ndarray,
    image_patches: np.ndarray,
    pos_t: np.ndarray,
    pos_hw: np.ndarray,
    attention_mask: np.ndarray,
    past_key_values: ort.OrtValue,
) -> tuple[np.ndarray, ort.OrtValue]:
    if session_outputs_token_id(session):
        return run_unified_kv_token_iobinding(
            session,
            tokens,
            image_patches,
            pos_t,
            pos_hw,
            attention_mask,
            past_key_values,
        )
    logits, present_key_values = run_unified_kv_iobinding(
        session,
        tokens,
        image_patches,
        pos_t,
        pos_hw,
        attention_mask,
        past_key_values,
    )
    next_token = np.argmax(logits, axis=-1).astype(np.int64)
    return next_token, present_key_values


def generate_with_onnx(
    kv_session: ort.InferenceSession,
    tokenizer,
    model_dir: Path,
    image_path: Path,
    category: str,
    *,
    max_new_tokens: int,
    min_dimension: int,
    max_dimension: int,
) -> dict:
    start = time.perf_counter()
    inputs, meta = build_model_inputs(
        model_dir,
        image_path,
        category,
        max_length=4096,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_sec = time.perf_counter() - start

    stop_token_ids = {meta["eos_token_id"], tokenizer.convert_tokens_to_ids("<|end_of_query|>")}
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    infer_start = time.perf_counter()
    empty_past = ortvalue_from_numpy_for_session(
        kv_session,
        np.empty(load_kv_cache_shape(model_dir, 1), dtype=np.float32),
    )
    next_token, past_key_values = run_next_token_iobinding(
        kv_session,
        inputs["tokens"].numpy().astype(np.int64),
        inputs["image_patches"].numpy().astype(np.float32),
        inputs["pos_t"].numpy().astype(np.int64),
        inputs["pos_hw"].numpy().astype(np.float32),
        inputs["attention_mask"].numpy().astype(np.bool_),
        empty_past,
    )
    current_length = int(meta["prompt_length"])
    current_pos_t = int(inputs["pos_t"][0, -1].item())
    decode_image_patch = np.zeros((1, 1, inputs["image_patches"].shape[-1]), dtype=np.float32)
    decode_pos_hw = np.full((1, 1, 2), np.nan, dtype=np.float32)
    for _ in range(max_new_tokens):
        token_id = int(next_token.reshape(-1)[0])
        generated_ids.append(token_id)
        if token_id in stop_token_ids:
            stop_reason = "stop_token"
            break
        token = np.array([[token_id]], dtype=np.int64)
        current_length += 1
        current_pos_t += 1
        pos_t = np.array([[current_pos_t]], dtype=np.int64)
        attention_mask = np.ones((1, 1, current_length), dtype=np.bool_)
        next_token, past_key_values = run_next_token_iobinding(
            kv_session,
            token,
            decode_image_patch,
            pos_t,
            decode_pos_hw,
            attention_mask,
            past_key_values,
        )
    inference_sec = time.perf_counter() - infer_start

    return {
        "ids": generated_ids,
        "text": decode_generated(tokenizer, generated_ids),
        "prompt_length": int(meta["prompt_length"]),
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason,
        "preprocess_sec": preprocess_sec,
        "inference_sec": inference_sec,
        "total_sec": preprocess_sec + inference_sec,
    }


def generate_batch_with_onnx(
    kv_session: ort.InferenceSession,
    tokenizer,
    model_dir: Path,
    image_paths: list[Path],
    categories: list[str],
    *,
    max_new_tokens: int,
    min_dimension: int,
    max_dimension: int,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    inputs, meta = build_batch_model_inputs(
        model_dir,
        image_paths,
        categories,
        max_length=4096,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    preprocess_sec = time.perf_counter() - start

    batch_size = int(meta["batch_size"])
    stop_token_ids = {meta["eos_token_id"], tokenizer.convert_tokens_to_ids("<|end_of_query|>")}
    generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    stop_reasons = ["max_new_tokens" for _ in range(batch_size)]
    finished = np.zeros((batch_size,), dtype=np.bool_)

    infer_start = time.perf_counter()
    empty_past = ortvalue_from_numpy_for_session(
        kv_session,
        np.empty(load_kv_cache_shape(model_dir, batch_size), dtype=np.float32),
    )
    next_token, past_key_values = run_next_token_iobinding(
        kv_session,
        inputs["tokens"].numpy().astype(np.int64),
        inputs["image_patches"].numpy().astype(np.float32),
        inputs["pos_t"].numpy().astype(np.int64),
        inputs["pos_hw"].numpy().astype(np.float32),
        inputs["attention_mask"].numpy().astype(np.bool_),
        empty_past,
    )

    current_length = int(meta["padded_prompt_length"])
    current_pos_t = inputs["pos_t"].amax(dim=1).numpy().astype(np.int64)
    pad_token_id = int(meta["pad_token_id"])
    current_attention_mask = (inputs["tokens"].numpy() != pad_token_id).astype(np.bool_)
    decode_image_patch = np.zeros(
        (batch_size, 1, inputs["image_patches"].shape[-1]), dtype=np.float32
    )
    decode_pos_hw = np.full((batch_size, 1, 2), np.nan, dtype=np.float32)

    for _ in range(max_new_tokens):
        was_finished = finished.copy()
        next_tokens = next_token.astype(np.int64).reshape(batch_size)
        next_tokens = np.where(was_finished, pad_token_id, next_tokens)

        for batch_idx, token_id in enumerate(next_tokens.tolist()):
            if finished[batch_idx]:
                continue
            generated_ids[batch_idx].append(int(token_id))
            if int(token_id) in stop_token_ids:
                finished[batch_idx] = True
                stop_reasons[batch_idx] = "stop_token"

        if bool(finished.all()):
            break

        token = next_tokens.reshape(batch_size, 1).astype(np.int64)
        current_length += 1
        current_pos_t = current_pos_t + 1
        pos_t = current_pos_t.reshape(batch_size, 1).astype(np.int64)
        current_attention_mask = np.concatenate(
            [current_attention_mask, (~was_finished).reshape(batch_size, 1)],
            axis=1,
        )
        attention_mask = current_attention_mask.reshape(batch_size, 1, current_length)
        next_token, past_key_values = run_next_token_iobinding(
            kv_session,
            token,
            decode_image_patch,
            pos_t,
            decode_pos_hw,
            attention_mask,
            past_key_values,
        )
    inference_sec = time.perf_counter() - infer_start

    per_sample_preprocess_sec = preprocess_sec / batch_size
    per_sample_inference_sec = inference_sec / batch_size
    results = []
    for ids, prompt_length, stop_reason in zip(
        generated_ids,
        meta["prompt_lengths"],
        stop_reasons,
    ):
        results.append(
            {
                "ids": ids,
                "text": decode_generated(tokenizer, ids),
                "prompt_length": int(prompt_length),
                "padded_prompt_length": int(meta["padded_prompt_length"]),
                "generated_tokens": len(ids),
                "stop_reason": stop_reason,
                "preprocess_sec": per_sample_preprocess_sec,
                "inference_sec": per_sample_inference_sec,
                "total_sec": per_sample_preprocess_sec + per_sample_inference_sec,
            }
        )

    batch_timing = {
        "batch_size": batch_size,
        "padded_prompt_length": int(meta["padded_prompt_length"]),
        "preprocess_sec": preprocess_sec,
        "inference_sec": inference_sec,
        "total_sec": preprocess_sec + inference_sec,
    }
    return results, batch_timing


def iter_batches(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        batch_size = len(items)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Falcon-OCR ONNX and Transformers inference.")
    parser.add_argument("--model-dir", type=Path, default=Path(r"D:\models\Falcon-OCR"))
    parser.add_argument("--kv-onnx", type=Path, default=Path(r"D:\models\Falcon-OCR-ONNX\falcon_ocr_kv_token.onnx"))
    parser.add_argument("--image-dir", type=Path, default=Path(r"D:\models\falcon-ocr-convert\imgs"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\models\Falcon-OCR-ONNX"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-dimension", type=int, default=64)
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--provider", default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            [p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

    print("Loading Transformers Falcon-OCR model in transformers5")
    transformers_model = load_transformers_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=True
    )

    print(f"Loading ONNX Runtime unified KV session: {args.kv_onnx}")
    kv_session = make_ort_session_with_provider(args.kv_onnx, args.provider)
    print(f"ONNX providers: {kv_session.get_providers()}")

    transformers_by_image = {}
    transformers_start = time.perf_counter()
    for idx, image_path in enumerate(image_paths, start=1):
        category = infer_category(image_path)
        print(f"[TF {idx}/{len(image_paths)}] {image_path.name} category={category}", flush=True)
        transformers_result = generate_with_transformers(
            transformers_model,
            image_path,
            category,
            max_new_tokens=args.max_new_tokens,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
        )
        transformers_by_image[image_path.name] = transformers_result
        print(
            f"  prompt={transformers_result['prompt_length']} "
            f"tokens={transformers_result['generated_tokens']} "
            f"tf_total={transformers_result['total_sec']:.3f}s",
            flush=True,
        )
    transformers_wall_sec = time.perf_counter() - transformers_start

    records = []
    batch_records = []
    onnx_wall_start = time.perf_counter()
    batches = iter_batches(image_paths, args.batch_size)
    for batch_idx, batch_paths in enumerate(batches, start=1):
        batch_categories = [infer_category(path) for path in batch_paths]
        print(
            f"[ONNX batch {batch_idx}/{len(batches)}] "
            f"size={len(batch_paths)} images={','.join(path.name for path in batch_paths)}",
            flush=True,
        )
        onnx_results, batch_timing = generate_batch_with_onnx(
            kv_session,
            tokenizer,
            args.model_dir,
            batch_paths,
            batch_categories,
            max_new_tokens=args.max_new_tokens,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
        )
        batch_records.append(
            {
                "batch_index": batch_idx,
                "images": [path.name for path in batch_paths],
                **batch_timing,
            }
        )
        for image_path, category, onnx_result in zip(batch_paths, batch_categories, onnx_results):
            transformers_result = transformers_by_image[image_path.name]
            tokens_equal = transformers_result["ids"] == onnx_result["ids"]
            text_equal = transformers_result["text"] == onnx_result["text"]
            record = {
                "image": image_path.name,
                "category": category,
                "prompt_length": transformers_result["prompt_length"],
                "onnx_padded_prompt_length": onnx_result["padded_prompt_length"],
                "transformers_text": transformers_result["text"],
                "onnx_text": onnx_result["text"],
                "tokens_equal": tokens_equal,
                "text_equal": text_equal,
                "transformers": transformers_result,
                "onnx": onnx_result,
                "speedup_total": (
                    transformers_result["total_sec"] / onnx_result["total_sec"]
                    if onnx_result["total_sec"] > 0
                    else None
                ),
                "speedup_inference": (
                    transformers_result["inference_sec"] / onnx_result["inference_sec"]
                    if onnx_result["inference_sec"] > 0
                    else None
                ),
            }
            records.append(record)
            print(
                f"  {image_path.name}: text_equal={text_equal} tokens_equal={tokens_equal} "
                f"onnx_tokens={onnx_result['generated_tokens']} "
                f"onnx_total_share={onnx_result['total_sec']:.3f}s",
                flush=True,
            )
    onnx_wall_sec = time.perf_counter() - onnx_wall_start

    for r in records:
        print(
            f"[OK] {r['image']} text_equal={r['text_equal']} tokens_equal={r['tokens_equal']} "
            f"tf_total={r['transformers']['total_sec']:.3f}s "
            f"onnx_total_share={r['onnx']['total_sec']:.3f}s",
            flush=True,
        )

    summary = {
        "model_dir": str(args.model_dir),
        "kv_onnx": str(args.kv_onnx),
        "kv_onnx_bytes": args.kv_onnx.stat().st_size,
        "image_dir": str(args.image_dir),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "onnx_providers": kv_session.get_providers(),
        "transformers_environment_note": (
            "transformers5 has CPU-only PyTorch; runtime compatibility patches are applied "
            "for Falcon-OCR remote code because triton/AuxRequest are unavailable."
        ),
        "all_text_equal": all(r["text_equal"] for r in records),
        "all_tokens_equal": all(r["tokens_equal"] for r in records),
        "onnx_batch_records": batch_records,
        "records": records,
    }
    tf_total = sum(r["transformers"]["total_sec"] for r in records)
    onnx_total = sum(b["total_sec"] for b in batch_records)
    summary["total_transformers_sec"] = tf_total
    summary["total_onnx_sec"] = onnx_total
    summary["overall_speedup_total"] = tf_total / onnx_total if onnx_total > 0 else None
    summary["transformers_wall_sec"] = transformers_wall_sec
    summary["onnx_batch_wall_sec"] = onnx_wall_sec

    json_path = args.output_dir / "onnx_vs_transformers_e2e.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "onnx_vs_transformers_perf.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "category",
                "prompt_length",
                "onnx_padded_prompt_length",
                "generated_tokens",
                "text_equal",
                "tokens_equal",
                "transformers_total_sec",
                "transformers_inference_sec",
                "onnx_total_sec",
                "onnx_inference_sec",
                "speedup_total",
                "speedup_inference",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "image": r["image"],
                    "category": r["category"],
                    "prompt_length": r["prompt_length"],
                    "onnx_padded_prompt_length": r["onnx_padded_prompt_length"],
                    "generated_tokens": r["transformers"]["generated_tokens"],
                    "text_equal": r["text_equal"],
                    "tokens_equal": r["tokens_equal"],
                    "transformers_total_sec": r["transformers"]["total_sec"],
                    "transformers_inference_sec": r["transformers"]["inference_sec"],
                    "onnx_total_sec": r["onnx"]["total_sec"],
                    "onnx_inference_sec": r["onnx"]["inference_sec"],
                    "speedup_total": r["speedup_total"],
                    "speedup_inference": r["speedup_inference"],
                }
            )

    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if not summary["all_text_equal"] or not summary["all_tokens_equal"]:
        raise SystemExit("ONNX and Transformers results differ")


if __name__ == "__main__":
    main()
