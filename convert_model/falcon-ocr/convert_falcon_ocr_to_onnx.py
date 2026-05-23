from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from falcon_ocr_onnx_model import (
    FalconOCRDecodeModel,
    FalconOCRExportConfig,
    FalconOCROnnxModel,
    FalconOCRPrefillModel,
    FalconOCRUnifiedKVModel,
    FalconOCRUnifiedKVTokenModel,
)


CATEGORY_PROMPTS = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}


def _import_model_module(model_dir: Path):
    sys.path.insert(0, str(model_dir.parent))
    module_name = model_dir.name
    return __import__(f"{module_name}.processing_falcon_ocr", fromlist=["process_batch"])


def load_export_config(model_dir: Path) -> FalconOCRExportConfig:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    return FalconOCRExportConfig(
        dim=int(data["dim"]),
        n_layers=int(data["n_layers"]),
        n_heads=int(data["n_heads"]),
        head_dim=int(data["head_dim"]),
        n_kv_heads=int(data["n_kv_heads"]),
        vocab_size=int(data["vocab_size"]),
        ffn_dim=int(data["ffn_dim"]),
        norm_eps=float(data["norm_eps"]),
        max_seq_len=int(data["max_seq_len"]),
        rope_theta=float(data["rope_theta"]),
        channel_size=int(data["channel_size"]),
        spatial_patch_size=int(data["spatial_patch_size"]),
        temporal_patch_size=int(data["temporal_patch_size"]),
        img_id=int(data["img_id"]),
    )


def load_export_model(model_dir: Path, dtype: torch.dtype = torch.float32) -> FalconOCROnnxModel:
    config = load_export_config(model_dir)
    model = FalconOCROnnxModel(config)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")

    converted_state = {}
    for key, tensor in state.items():
        converted_key = key
        if converted_key.startswith("layers."):
            parts = converted_key.split(".")
            converted_key = ".".join(["layers", parts[1], *parts[2:]])
        converted_state[converted_key] = tensor

    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, unexpected={unexpected}")
    model.to(dtype=dtype)
    model.eval()
    return model


def make_attention_mask(
    tokens: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int,
    soi_token_id: int,
    eoi_token_id: int,
) -> torch.Tensor:
    batch, seq = tokens.shape
    device = tokens.device
    positions = torch.arange(seq, device=device)
    q_idx = positions.view(1, seq, 1)
    kv_idx = positions.view(1, 1, seq)

    causal = q_idx >= kv_idx

    eos_mask = tokens == eos_token_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int64)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]
    document = sequence_indices[:, :, None] == sequence_indices[:, None, :]

    non_pad_mask_id = torch.cumsum(tokens != pad_token_id, dim=1)
    non_left_pad = non_pad_mask_id[:, None, :] > 0

    block_causal = causal & document & non_left_pad

    soi_mask = tokens == soi_token_id
    eoi_mask = tokens == eoi_token_id
    acc_soi_mask = torch.cumsum(soi_mask.to(torch.int64), dim=1)
    acc_eoi_mask = torch.cumsum(eoi_mask.to(torch.int64), dim=1)
    img_mask = (acc_soi_mask - acc_eoi_mask) > 0
    img_indices = acc_soi_mask * img_mask.to(torch.int64)
    image_prefix = (
        img_mask[:, :, None]
        & img_mask[:, None, :]
        & (img_indices[:, :, None] == img_indices[:, None, :])
    )

    return image_prefix | block_causal


def build_model_inputs(
    model_dir: Path,
    image_path: Path,
    category: str,
    *,
    max_length: int,
    min_dimension: int,
    max_dimension: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    processing = _import_model_module(model_dir)
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = type("Config", (), config_data)()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    prompt = f"<|image|>{CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS['plain'])}\n<|OCR_PLAIN|>"
    batch = processing.process_batch(
        tokenizer,
        config,
        [(str(image_path), prompt)],
        max_length=max_length,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )

    tokens = batch["tokens"].to(torch.long)
    pixel_values = batch["pixel_values"].to(torch.float32)
    pixel_mask = batch["pixel_mask"].to(torch.bool)
    pos_t = batch["pos_t"].to(torch.long)
    pos_hw = batch["pos_hw"].to(torch.float32)

    patch = int(config.spatial_patch_size)
    temporal = int(config.temporal_patch_size)
    image_patches_all = (
        pixel_values.reshape(
            pixel_values.shape[0],
            pixel_values.shape[1] // temporal,
            temporal,
            pixel_values.shape[2] // patch,
            patch,
            pixel_values.shape[3] // patch,
            patch,
            pixel_values.shape[4],
        )
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(pixel_values.shape[0], -1, temporal * patch * patch * pixel_values.shape[4])
    )
    patch_mask = (
        pixel_mask.reshape(
            pixel_mask.shape[0],
            pixel_mask.shape[1] // temporal,
            temporal,
            pixel_mask.shape[2] // patch,
            patch,
            pixel_mask.shape[3] // patch,
            patch,
        )
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(pixel_mask.shape[0], -1, temporal * patch * patch)
        .any(dim=-1)
    )
    valid_patches = image_patches_all[patch_mask]

    image_patch_dim = valid_patches.shape[-1]
    image_patches = torch.zeros(
        (tokens.shape[0], tokens.shape[1], image_patch_dim), dtype=torch.float32
    )
    img_mask = tokens == int(config.img_id)
    if int(img_mask.sum().item()) != valid_patches.shape[0]:
        raise RuntimeError(
            f"Image token count mismatch: tokens={int(img_mask.sum().item())}, "
            f"patches={valid_patches.shape[0]}"
        )
    image_patches[img_mask] = valid_patches

    attention_mask = make_attention_mask(
        tokens,
        pad_token_id=pad_token_id,
        eos_token_id=int(config.eos_id),
        soi_token_id=int(config.image_cls_token_id),
        eoi_token_id=int(config.img_end_id),
    )
    meta = {
        "pad_token_id": int(pad_token_id),
        "eos_token_id": int(config.eos_id),
        "soi_token_id": int(config.image_cls_token_id),
        "eoi_token_id": int(config.img_end_id),
        "img_id": int(config.img_id),
        "prompt_length": int(tokens.shape[1]),
    }
    inputs = {
        "tokens": tokens,
        "image_patches": image_patches,
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "attention_mask": attention_mask,
    }
    return inputs, meta


def build_batch_model_inputs(
    model_dir: Path,
    image_paths: list[Path],
    categories: list[str],
    *,
    max_length: int,
    min_dimension: int,
    max_dimension: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    if len(image_paths) != len(categories):
        raise ValueError("image_paths and categories must have the same length")
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    processing = _import_model_module(model_dir)
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = type("Config", (), config_data)()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    image_prompt_pairs = [
        (
            str(image_path),
            f"<|image|>{CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS['plain'])}\n<|OCR_PLAIN|>",
        )
        for image_path, category in zip(image_paths, categories)
    ]
    batch = processing.process_batch(
        tokenizer,
        config,
        image_prompt_pairs,
        max_length=max_length,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )

    tokens = batch["tokens"].to(torch.long)
    pixel_values = batch["pixel_values"].to(torch.float32)
    pixel_mask = batch["pixel_mask"].to(torch.bool)
    pos_t = batch["pos_t"].to(torch.long)
    pos_hw = batch["pos_hw"].to(torch.float32)

    patch = int(config.spatial_patch_size)
    temporal = int(config.temporal_patch_size)
    image_patches_all = (
        pixel_values.reshape(
            pixel_values.shape[0],
            pixel_values.shape[1] // temporal,
            temporal,
            pixel_values.shape[2] // patch,
            patch,
            pixel_values.shape[3] // patch,
            patch,
            pixel_values.shape[4],
        )
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(pixel_values.shape[0], -1, temporal * patch * patch * pixel_values.shape[4])
    )
    patch_mask = (
        pixel_mask.reshape(
            pixel_mask.shape[0],
            pixel_mask.shape[1] // temporal,
            temporal,
            pixel_mask.shape[2] // patch,
            patch,
            pixel_mask.shape[3] // patch,
            patch,
        )
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(pixel_mask.shape[0], -1, temporal * patch * patch)
        .any(dim=-1)
    )
    valid_patches = image_patches_all[patch_mask]

    image_patch_dim = valid_patches.shape[-1]
    image_patches = torch.zeros(
        (tokens.shape[0], tokens.shape[1], image_patch_dim), dtype=torch.float32
    )
    img_mask = tokens == int(config.img_id)
    if int(img_mask.sum().item()) != valid_patches.shape[0]:
        raise RuntimeError(
            f"Image token count mismatch: tokens={int(img_mask.sum().item())}, "
            f"patches={valid_patches.shape[0]}"
        )
    image_patches[img_mask] = valid_patches

    attention_mask = make_attention_mask(
        tokens,
        pad_token_id=pad_token_id,
        eos_token_id=int(config.eos_id),
        soi_token_id=int(config.image_cls_token_id),
        eoi_token_id=int(config.img_end_id),
    )
    prompt_lengths = (tokens != pad_token_id).sum(dim=1).to(torch.long)
    meta = {
        "pad_token_id": int(pad_token_id),
        "eos_token_id": int(config.eos_id),
        "soi_token_id": int(config.image_cls_token_id),
        "eoi_token_id": int(config.img_end_id),
        "img_id": int(config.img_id),
        "batch_size": int(tokens.shape[0]),
        "padded_prompt_length": int(tokens.shape[1]),
        "prompt_lengths": [int(x) for x in prompt_lengths.tolist()],
    }
    inputs = {
        "tokens": tokens,
        "image_patches": image_patches,
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "attention_mask": attention_mask,
    }
    return inputs, meta


def append_token_inputs(
    inputs: dict[str, torch.Tensor],
    token_id: int,
    *,
    meta: dict[str, int],
) -> dict[str, torch.Tensor]:
    tokens = torch.cat(
        [inputs["tokens"], torch.tensor([[token_id]], dtype=torch.long)],
        dim=1,
    )
    image_patches = torch.cat(
        [
            inputs["image_patches"],
            torch.zeros(
                (1, 1, inputs["image_patches"].shape[-1]),
                dtype=inputs["image_patches"].dtype,
            ),
        ],
        dim=1,
    )
    pos_t = torch.cat(
        [inputs["pos_t"], inputs["pos_t"][:, -1:] + 1],
        dim=1,
    )
    pos_hw = torch.cat(
        [
            inputs["pos_hw"],
            torch.full((1, 1, 2), float("nan"), dtype=inputs["pos_hw"].dtype),
        ],
        dim=1,
    )
    attention_mask = make_attention_mask(
        tokens,
        pad_token_id=meta["pad_token_id"],
        eos_token_id=meta["eos_token_id"],
        soi_token_id=meta["soi_token_id"],
        eoi_token_id=meta["eoi_token_id"],
    )
    return {
        "tokens": tokens,
        "image_patches": image_patches,
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "attention_mask": attention_mask,
    }


def build_full_length_inputs(
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    *,
    target_length: int,
    fill_token_id: int,
) -> dict[str, torch.Tensor]:
    prompt_length = int(initial_inputs["tokens"].shape[1])
    if target_length < prompt_length:
        raise ValueError(
            f"target_length={target_length} is shorter than prompt_length={prompt_length}"
        )
    if target_length == prompt_length:
        return {k: v.clone() for k, v in initial_inputs.items()}

    tail_length = target_length - prompt_length
    tail_tokens = torch.full((1, tail_length), fill_token_id, dtype=torch.long)
    tokens = torch.cat([initial_inputs["tokens"], tail_tokens], dim=1)

    tail_patches = torch.zeros(
        (1, tail_length, initial_inputs["image_patches"].shape[-1]),
        dtype=initial_inputs["image_patches"].dtype,
    )
    image_patches = torch.cat([initial_inputs["image_patches"], tail_patches], dim=1)

    start_pos = initial_inputs["pos_t"][:, -1:] + 1
    tail_pos_t = start_pos + torch.arange(tail_length, dtype=torch.long).view(1, -1)
    pos_t = torch.cat([initial_inputs["pos_t"], tail_pos_t], dim=1)

    tail_pos_hw = torch.full(
        (1, tail_length, 2),
        float("nan"),
        dtype=initial_inputs["pos_hw"].dtype,
    )
    pos_hw = torch.cat([initial_inputs["pos_hw"], tail_pos_hw], dim=1)

    attention_mask = make_attention_mask(
        tokens,
        pad_token_id=meta["pad_token_id"],
        eos_token_id=meta["eos_token_id"],
        soi_token_id=meta["soi_token_id"],
        eoi_token_id=meta["eoi_token_id"],
    )
    return {
        "tokens": tokens,
        "image_patches": image_patches,
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "attention_mask": attention_mask,
    }


def build_inputs_from_tokens(
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prompt_length = int(initial_inputs["tokens"].shape[1])
    target_length = int(tokens.shape[1])
    if target_length < prompt_length:
        raise ValueError(
            f"target_length={target_length} is shorter than prompt_length={prompt_length}"
        )

    tail_length = target_length - prompt_length
    tail_patches = torch.zeros(
        (1, tail_length, initial_inputs["image_patches"].shape[-1]),
        dtype=initial_inputs["image_patches"].dtype,
    )
    image_patches = torch.cat([initial_inputs["image_patches"], tail_patches], dim=1)

    if tail_length:
        start_pos = initial_inputs["pos_t"][:, -1:] + 1
        tail_pos_t = start_pos + torch.arange(tail_length, dtype=torch.long).view(1, -1)
        tail_pos_hw = torch.full(
            (1, tail_length, 2),
            float("nan"),
            dtype=initial_inputs["pos_hw"].dtype,
        )
        pos_t = torch.cat([initial_inputs["pos_t"], tail_pos_t], dim=1)
        pos_hw = torch.cat([initial_inputs["pos_hw"], tail_pos_hw], dim=1)
    else:
        pos_t = initial_inputs["pos_t"].clone()
        pos_hw = initial_inputs["pos_hw"].clone()

    attention_mask = make_attention_mask(
        tokens,
        pad_token_id=meta["pad_token_id"],
        eos_token_id=meta["eos_token_id"],
        soi_token_id=meta["soi_token_id"],
        eoi_token_id=meta["eoi_token_id"],
    )
    return {
        "tokens": tokens,
        "image_patches": image_patches,
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "attention_mask": attention_mask,
    }


def topk_from_logits(logits: torch.Tensor | np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits)
    else:
        logits_t = logits.detach().cpu()
    values, indices = torch.topk(logits_t.float(), k=k, dim=-1)
    return indices.numpy(), values.numpy()


def run_pytorch(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
        )


def make_ort_session(onnx_path: Path) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def run_ort(session: ort.InferenceSession, inputs: dict[str, torch.Tensor]) -> np.ndarray:
    feed = {
        "tokens": inputs["tokens"].numpy().astype(np.int64),
        "image_patches": inputs["image_patches"].numpy().astype(np.float32),
        "pos_t": inputs["pos_t"].numpy().astype(np.int64),
        "pos_hw": inputs["pos_hw"].numpy().astype(np.float32),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.bool_),
    }
    return session.run(["logits"], feed)[0]


def run_ort_kv_prefill(
    session: ort.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    feed = {
        "tokens": inputs["tokens"].numpy().astype(np.int64),
        "image_patches": inputs["image_patches"].numpy().astype(np.float32),
        "pos_t": inputs["pos_t"].numpy().astype(np.int64),
        "pos_hw": inputs["pos_hw"].numpy().astype(np.float32),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.bool_),
    }
    logits, present = session.run(["logits", "present_key_values"], feed)
    return logits, present


def run_ort_kv_decode(
    session: ort.InferenceSession,
    token: torch.Tensor,
    pos_t: torch.Tensor,
    pos_hw: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feed = {
        "token": token.numpy().astype(np.int64),
        "pos_t": pos_t.numpy().astype(np.int64),
        "pos_hw": pos_hw.numpy().astype(np.float32),
        "attention_mask": attention_mask.numpy().astype(np.bool_),
        "past_key_values": past_key_values.astype(np.float32),
    }
    logits, present = session.run(["logits", "present_key_values"], feed)
    return logits, present


def run_ort_unified_kv(
    session: ort.InferenceSession,
    tokens: torch.Tensor,
    image_patches: torch.Tensor,
    pos_t: torch.Tensor,
    pos_hw: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feed = {
        "tokens": tokens.numpy().astype(np.int64),
        "image_patches": image_patches.numpy().astype(np.float32),
        "pos_t": pos_t.numpy().astype(np.int64),
        "pos_hw": pos_hw.numpy().astype(np.float32),
        "attention_mask": attention_mask.numpy().astype(np.bool_),
        "past_key_values": past_key_values.astype(np.float32),
    }
    logits, present = session.run(["logits", "present_key_values"], feed)
    return logits, present


def run_ort_unified_kv_token(
    session: ort.InferenceSession,
    tokens: torch.Tensor,
    image_patches: torch.Tensor,
    pos_t: torch.Tensor,
    pos_hw: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feed = {
        "tokens": tokens.numpy().astype(np.int64),
        "image_patches": image_patches.numpy().astype(np.float32),
        "pos_t": pos_t.numpy().astype(np.int64),
        "pos_hw": pos_hw.numpy().astype(np.float32),
        "attention_mask": attention_mask.numpy().astype(np.bool_),
        "past_key_values": past_key_values.astype(np.float32),
    }
    next_token, present = session.run(["next_token", "present_key_values"], feed)
    return next_token, present


def choose_fill_token(model: FalconOCROnnxModel, inputs: dict[str, torch.Tensor]) -> int:
    logits = run_pytorch(model, inputs)[:, -1, :]
    return int(torch.argmax(logits, dim=-1).item())


def _attention_output(
    attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
    mask_value = torch.tensor(-3.4028234663852886e38, dtype=scores.dtype, device=scores.device)
    scores = torch.where(attention_mask.unsqueeze(1), scores, mask_value)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    sink_scale = torch.sigmoid(lse - attention.sinks.reshape(1, -1, 1))
    output = output * sink_scale.unsqueeze(-1)
    output = output.permute(0, 2, 1, 3).contiguous().flatten(2)
    return attention.wo(output)


def _make_decode_attention_row(tokens: torch.Tensor, meta: dict[str, int]) -> torch.Tensor:
    eos_mask = tokens == meta["eos_token_id"]
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int64)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]
    document_mask = sequence_indices[:, -1:] == sequence_indices
    non_pad_mask = torch.cumsum(tokens != meta["pad_token_id"], dim=1) > 0
    return (document_mask & non_pad_mask).unsqueeze(1)


def make_decode_inputs(
    token_id: int,
    current_length: int,
    *,
    meta: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token = torch.tensor([[token_id]], dtype=torch.long)
    pos_t = torch.tensor([[current_length]], dtype=torch.long)
    pos_hw = torch.full((1, 1, 2), float("nan"), dtype=torch.float32)
    tokens = torch.full((1, current_length + 1), meta["pad_token_id"], dtype=torch.long)
    tokens[:, :current_length] = 1
    tokens[:, current_length] = token_id
    attention_mask = torch.ones((1, 1, current_length + 1), dtype=torch.bool)
    return token, pos_t, pos_hw, attention_mask


def _prefill_incremental(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    tokens = inputs["tokens"]
    h = model.tok_embeddings(tokens)
    image_features = model.img_projector(inputs["image_patches"])
    h = torch.where((tokens == model.config.img_id).unsqueeze(-1), image_features, h)
    rope_cos, rope_sin, spatial_cos, spatial_sin = model._rope_inputs(
        inputs["pos_t"], inputs["pos_hw"]
    )

    caches = []
    for layer in model.layers:
        attention = layer.attention
        xq, xk, xv = attention._pre_attention_qkv(h)
        xq, xk = attention._apply_rope(
            xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin
        )
        q = xq.permute(0, 2, 1, 3)
        k = xk.permute(0, 2, 1, 3).contiguous()
        v = xv.permute(0, 2, 1, 3).contiguous()
        h = h + _attention_output(attention, q, k, v, inputs["attention_mask"])
        h = h + layer.feed_forward(h)
        caches.append([k, v])

    logits = model.output(model.norm(h[:, -1:, :]))[:, 0, :]
    return caches, logits


def _decode_incremental(
    model: FalconOCROnnxModel,
    caches: list[list[torch.Tensor]],
    token: torch.Tensor,
    pos_t: torch.Tensor,
    attention_row: torch.Tensor,
) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    h = model.tok_embeddings(token)
    pos_hw = torch.full((1, 1, 2), float("nan"), dtype=torch.float32)
    rope_cos, rope_sin, spatial_cos, spatial_sin = model._rope_inputs(pos_t, pos_hw)

    new_caches = []
    for layer, (cached_k, cached_v) in zip(model.layers, caches):
        attention = layer.attention
        xq, xk, xv = attention._pre_attention_qkv(h)
        xq, xk = attention._apply_rope(
            xq, xk, rope_cos, rope_sin, spatial_cos, spatial_sin
        )
        q = xq.permute(0, 2, 1, 3)
        new_k = xk.permute(0, 2, 1, 3).contiguous()
        new_v = xv.permute(0, 2, 1, 3).contiguous()
        k = torch.cat([cached_k, new_k], dim=2)
        v = torch.cat([cached_v, new_v], dim=2)
        h = h + _attention_output(attention, q, k, v, attention_row)
        h = h + layer.feed_forward(h)
        new_caches.append([k, v])

    logits = model.output(model.norm(h))[:, 0, :]
    return new_caches, logits


def greedy_generate_pytorch_logits(
    model: FalconOCROnnxModel,
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    *,
    target_length: int,
    progress_interval: int = 100,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt_length = int(initial_inputs["tokens"].shape[1])
    if target_length < prompt_length:
        raise ValueError(
            f"target_length={target_length} is shorter than prompt_length={prompt_length}"
        )

    positions_compared = target_length - prompt_length + 1
    vocab_size = model.config.vocab_size
    compared_logits = torch.empty((positions_compared, vocab_size), dtype=torch.float32)
    tokens = initial_inputs["tokens"].clone()
    pos_t = initial_inputs["pos_t"][:, -1:].clone()

    print(f"Running PyTorch greedy prefill at prompt length {prompt_length}", flush=True)
    with torch.inference_mode():
        caches, logits = _prefill_incremental(model, initial_inputs)
        compared_logits[0].copy_(logits[0].float())

        for step in range(1, positions_compared):
            next_token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)
            tokens = torch.cat([tokens, next_token], dim=1)
            pos_t = pos_t + 1
            attention_row = _make_decode_attention_row(tokens, meta)
            caches, logits = _decode_incremental(model, caches, next_token, pos_t, attention_row)
            compared_logits[step].copy_(logits[0].float())
            if progress_interval and (step % progress_interval == 0 or step == positions_compared - 1):
                print(
                    f"PyTorch greedy step {step}/{positions_compared - 1}; "
                    f"sequence length {tokens.shape[1]}",
                    flush=True,
                )

    full_inputs = build_inputs_from_tokens(initial_inputs, meta, tokens)
    return full_inputs, compared_logits


def validate_top5(
    model: FalconOCROnnxModel,
    session: ort.InferenceSession,
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    *,
    steps: int,
) -> dict:
    inputs = {k: v.clone() for k, v in initial_inputs.items()}
    records = []
    all_match = True
    max_abs_diff = 0.0

    for step in range(steps):
        pt_logits = run_pytorch(model, inputs)[:, -1, :]
        ort_logits = run_ort(session, inputs)[:, -1, :]
        pt_top_idx, pt_top_vals = topk_from_logits(pt_logits, 5)
        ort_top_idx, ort_top_vals = topk_from_logits(ort_logits, 5)
        step_abs_diff = float(np.max(np.abs(pt_logits.detach().cpu().numpy() - ort_logits)))
        top5_match = bool(np.array_equal(pt_top_idx, ort_top_idx))
        all_match = all_match and top5_match
        max_abs_diff = max(max_abs_diff, step_abs_diff)

        token_id = int(pt_top_idx[0, 0])
        records.append(
            {
                "step": step,
                "sequence_length": int(inputs["tokens"].shape[1]),
                "next_token_id": token_id,
                "top5_match": top5_match,
                "max_abs_diff": step_abs_diff,
                "pytorch_top5_ids": pt_top_idx[0].astype(int).tolist(),
                "pytorch_top5_logits": [float(x) for x in pt_top_vals[0].tolist()],
                "onnx_top5_ids": ort_top_idx[0].astype(int).tolist(),
                "onnx_top5_logits": [float(x) for x in ort_top_vals[0].tolist()],
            }
        )

        inputs = append_token_inputs(inputs, token_id, meta=meta)

    return {
        "steps": steps,
        "all_top5_match": all_match,
        "max_abs_diff": max_abs_diff,
        "records": records,
    }


def validate_full_length_top5(
    model: FalconOCROnnxModel,
    session: ort.InferenceSession,
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    *,
    target_length: int,
    fill_token_id: int | None,
    chunk_size: int = 32,
) -> dict:
    if fill_token_id is None:
        fill_token_id = choose_fill_token(model, initial_inputs)

    prompt_length = int(initial_inputs["tokens"].shape[1])
    full_inputs = build_full_length_inputs(
        initial_inputs,
        meta,
        target_length=target_length,
        fill_token_id=fill_token_id,
    )

    print(f"Running PyTorch full forward at sequence length {target_length}")
    pt_logits = run_pytorch(model, full_inputs).detach().cpu()
    print(f"Running ONNX Runtime full forward at sequence length {target_length}")
    ort_logits = run_ort(session, full_inputs)

    compare_start = prompt_length - 1
    compare_end = target_length - 1
    records = []
    all_match = True
    max_abs_diff = 0.0
    mismatches = []

    for start in range(compare_start, compare_end + 1, chunk_size):
        end = min(start + chunk_size, compare_end + 1)
        pt_chunk = pt_logits[0, start:end].float()
        ort_chunk = torch.from_numpy(ort_logits[0, start:end]).float()

        chunk_abs_diff = torch.max(torch.abs(pt_chunk - ort_chunk)).item()
        max_abs_diff = max(max_abs_diff, float(chunk_abs_diff))

        pt_vals, pt_idx = torch.topk(pt_chunk, k=5, dim=-1)
        ort_vals, ort_idx = torch.topk(ort_chunk, k=5, dim=-1)

        for offset in range(end - start):
            position = start + offset
            step = position - compare_start
            top5_match = bool(torch.equal(pt_idx[offset], ort_idx[offset]))
            if not top5_match:
                all_match = False
                mismatches.append(step)

            records.append(
                {
                    "step": int(step),
                    "logit_position": int(position),
                    "sequence_length_for_step": int(position + 1),
                    "top5_match": top5_match,
                    "pytorch_top5_ids": [int(x) for x in pt_idx[offset].tolist()],
                    "pytorch_top5_logits": [float(x) for x in pt_vals[offset].tolist()],
                    "onnx_top5_ids": [int(x) for x in ort_idx[offset].tolist()],
                    "onnx_top5_logits": [float(x) for x in ort_vals[offset].tolist()],
                }
            )

    return {
        "mode": "full_length",
        "target_length": int(target_length),
        "onnx_input_shapes": {k: list(v.shape) for k, v in full_inputs.items()},
        "prompt_length": int(prompt_length),
        "fill_token_id": int(fill_token_id),
        "compare_start_position": int(compare_start),
        "compare_end_position": int(compare_end),
        "positions_compared": int(compare_end - compare_start + 1),
        "generated_steps_to_max_length": int(max(target_length - prompt_length, 0)),
        "all_top5_match": all_match,
        "mismatch_count": int(len(mismatches)),
        "mismatch_steps": [int(x) for x in mismatches[:1000]],
        "max_abs_diff": max_abs_diff,
        "records": records,
    }


def validate_full_length_greedy_top5(
    model: FalconOCROnnxModel,
    session: ort.InferenceSession,
    initial_inputs: dict[str, torch.Tensor],
    meta: dict[str, int],
    *,
    target_length: int,
    chunk_size: int = 32,
    progress_interval: int = 100,
) -> dict:
    prompt_length = int(initial_inputs["tokens"].shape[1])
    compare_start = prompt_length - 1
    compare_end = target_length - 1

    full_inputs, pt_logits = greedy_generate_pytorch_logits(
        model,
        initial_inputs,
        meta,
        target_length=target_length,
        progress_interval=progress_interval,
    )

    print(f"Running ONNX Runtime full forward at sequence length {target_length}", flush=True)
    ort_logits = run_ort(session, full_inputs)
    ort_compare = ort_logits[0, compare_start : compare_end + 1]

    records = []
    all_match = True
    max_abs_diff = 0.0
    mismatches = []

    for start in range(0, pt_logits.shape[0], chunk_size):
        end = min(start + chunk_size, pt_logits.shape[0])
        pt_chunk = pt_logits[start:end].float()
        ort_chunk = torch.from_numpy(ort_compare[start:end]).float()

        chunk_abs_diff = torch.max(torch.abs(pt_chunk - ort_chunk)).item()
        max_abs_diff = max(max_abs_diff, float(chunk_abs_diff))

        pt_vals, pt_idx = torch.topk(pt_chunk, k=5, dim=-1)
        ort_vals, ort_idx = torch.topk(ort_chunk, k=5, dim=-1)

        for offset in range(end - start):
            step = start + offset
            position = compare_start + step
            top5_match = bool(torch.equal(pt_idx[offset], ort_idx[offset]))
            if not top5_match:
                all_match = False
                mismatches.append(step)

            records.append(
                {
                    "step": int(step),
                    "logit_position": int(position),
                    "sequence_length_for_step": int(position + 1),
                    "greedy_next_token_id": int(pt_idx[offset, 0].item()),
                    "top5_match": top5_match,
                    "pytorch_top5_ids": [int(x) for x in pt_idx[offset].tolist()],
                    "pytorch_top5_logits": [float(x) for x in pt_vals[offset].tolist()],
                    "onnx_top5_ids": [int(x) for x in ort_idx[offset].tolist()],
                    "onnx_top5_logits": [float(x) for x in ort_vals[offset].tolist()],
                }
            )

    return {
        "mode": "full_length_greedy",
        "target_length": int(target_length),
        "onnx_input_shapes": {k: list(v.shape) for k, v in full_inputs.items()},
        "prompt_length": int(prompt_length),
        "compare_start_position": int(compare_start),
        "compare_end_position": int(compare_end),
        "positions_compared": int(compare_end - compare_start + 1),
        "generated_steps_to_max_length": int(max(target_length - prompt_length, 0)),
        "all_top5_match": all_match,
        "mismatch_count": int(len(mismatches)),
        "mismatch_steps": [int(x) for x in mismatches[:1000]],
        "max_abs_diff": max_abs_diff,
        "records": records,
    }


def export_onnx(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
    output_path: Path,
    *,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
        ),
        str(output_path),
        input_names=["tokens", "image_patches", "pos_t", "pos_hw", "attention_mask"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={
            "tokens": {0: "batch", 1: "sequence"},
            "image_patches": {0: "batch", 1: "sequence"},
            "pos_t": {0: "batch", 1: "sequence"},
            "pos_hw": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence", 2: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        do_constant_folding=True,
        external_data=True,
    )
    model_proto = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(model_proto)


def export_kv_onnx(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    opset: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = output_dir / "falcon_ocr_prefill.onnx"
    decode_path = output_dir / "falcon_ocr_decode.onnx"

    prefill = FalconOCRPrefillModel(model).eval()
    torch.onnx.export(
        prefill,
        (
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
        ),
        str(prefill_path),
        input_names=["tokens", "image_patches", "pos_t", "pos_hw", "attention_mask"],
        output_names=["logits", "present_key_values"],
        opset_version=opset,
        dynamic_axes={
            "tokens": {0: "batch", 1: "sequence"},
            "image_patches": {0: "batch", 1: "sequence"},
            "pos_t": {0: "batch", 1: "sequence"},
            "pos_hw": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence", 2: "sequence"},
            "present_key_values": {4: "total_sequence"},
        },
        do_constant_folding=True,
        external_data=True,
    )

    prompt_len = int(inputs["tokens"].shape[1])
    with torch.inference_mode():
        _, past = prefill(
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
        )
    token = torch.tensor([[571]], dtype=torch.long)
    pos_t = torch.tensor([[prompt_len]], dtype=torch.long)
    pos_hw = torch.full((1, 1, 2), float("nan"), dtype=torch.float32)
    attention_mask = torch.ones((1, 1, prompt_len + 1), dtype=torch.bool)

    decode = FalconOCRDecodeModel(model).eval()
    torch.onnx.export(
        decode,
        (token, pos_t, pos_hw, attention_mask, past),
        str(decode_path),
        input_names=["token", "pos_t", "pos_hw", "attention_mask", "past_key_values"],
        output_names=["logits", "present_key_values"],
        opset_version=opset,
        dynamic_axes={
            "token": {0: "batch"},
            "pos_t": {0: "batch"},
            "pos_hw": {0: "batch"},
            "attention_mask": {0: "batch", 2: "total_sequence"},
            "past_key_values": {4: "past_sequence"},
            "present_key_values": {4: "total_sequence"},
        },
        do_constant_folding=True,
        external_data=True,
    )

    for path in (prefill_path, decode_path):
        model_proto = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model_proto)
    return prefill_path, decode_path


def export_unified_kv_onnx(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    opset: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    kv_path = output_dir / "falcon_ocr_kv.onnx"
    kv_model = FalconOCRUnifiedKVModel(model).eval()
    empty_past = torch.empty(
        (
            model.config.n_layers,
            2,
            1,
            model.config.n_heads,
            0,
            model.config.head_dim,
        ),
        dtype=torch.float32,
    )
    torch.onnx.export(
        kv_model,
        (
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
            empty_past,
        ),
        str(kv_path),
        input_names=[
            "tokens",
            "image_patches",
            "pos_t",
            "pos_hw",
            "attention_mask",
            "past_key_values",
        ],
        output_names=["logits", "present_key_values"],
        opset_version=opset,
        dynamic_axes={
            "tokens": {0: "batch", 1: "sequence"},
            "image_patches": {0: "batch", 1: "sequence"},
            "pos_t": {0: "batch", 1: "sequence"},
            "pos_hw": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence", 2: "total_sequence"},
            "past_key_values": {2: "batch", 4: "past_sequence"},
            "present_key_values": {2: "batch", 4: "total_sequence"},
        },
        do_constant_folding=True,
        external_data=True,
    )
    model_proto = onnx.load(str(kv_path), load_external_data=False)
    onnx.checker.check_model(model_proto)
    return kv_path


def export_unified_kv_token_onnx(
    model: FalconOCROnnxModel,
    inputs: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    opset: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    kv_path = output_dir / "falcon_ocr_kv_token.onnx"
    kv_model = FalconOCRUnifiedKVTokenModel(model).eval()
    empty_past = torch.empty(
        (
            model.config.n_layers,
            2,
            1,
            model.config.n_heads,
            0,
            model.config.head_dim,
        ),
        dtype=torch.float32,
    )
    torch.onnx.export(
        kv_model,
        (
            inputs["tokens"],
            inputs["image_patches"],
            inputs["pos_t"],
            inputs["pos_hw"],
            inputs["attention_mask"],
            empty_past,
        ),
        str(kv_path),
        input_names=[
            "tokens",
            "image_patches",
            "pos_t",
            "pos_hw",
            "attention_mask",
            "past_key_values",
        ],
        output_names=["next_token", "present_key_values"],
        opset_version=opset,
        dynamic_axes={
            "tokens": {0: "batch", 1: "sequence"},
            "image_patches": {0: "batch", 1: "sequence"},
            "pos_t": {0: "batch", 1: "sequence"},
            "pos_hw": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence", 2: "total_sequence"},
            "past_key_values": {2: "batch", 4: "past_sequence"},
            "next_token": {0: "batch"},
            "present_key_values": {2: "batch", 4: "total_sequence"},
        },
        do_constant_folding=True,
        external_data=True,
    )
    model_proto = onnx.load(str(kv_path), load_external_data=False)
    onnx.checker.check_model(model_proto)
    return kv_path


def copy_runtime_files(model_dir: Path, output_dir: Path) -> None:
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processing_falcon_ocr.py",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Falcon-OCR weights to ONNX.")
    parser.add_argument("--model-dir", type=Path, default=Path(r"D:\models\Falcon-OCR"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\models\Falcon-OCR-ONNX"))
    parser.add_argument("--image", type=Path, default=Path(r"D:\models\falcon-ocr-convert\imgs\text-2026-05-23-124542.png"))
    parser.add_argument("--category", default="text", choices=sorted(CATEGORY_PROMPTS))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--validation-mode", choices=["sample", "full-length"], default="full-length")
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--fill-token-id", type=int, default=None)
    parser.add_argument("--compare-chunk-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--full-length-generation", choices=["greedy", "fill"], default="greedy")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-dimension", type=int, default=64)
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--report-name", default="top5_validation.json")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--export-kv", action="store_true")
    parser.add_argument("--export-unified-kv", action="store_true")
    parser.add_argument("--export-unified-kv-token", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "falcon_ocr.onnx"

    print(f"Loading Falcon-OCR from {args.model_dir}")
    model = load_export_model(args.model_dir, dtype=torch.float32)
    inputs, meta = build_model_inputs(
        args.model_dir,
        args.image,
        args.category,
        max_length=args.max_length,
        min_dimension=args.min_dimension,
        max_dimension=args.max_dimension,
    )
    print(
        "Prepared inputs:",
        f"tokens={tuple(inputs['tokens'].shape)}",
        f"image_patches={tuple(inputs['image_patches'].shape)}",
    )

    if args.export_unified_kv_token:
        print(f"Exporting token-only unified KV-cache ONNX model to {args.output_dir}")
        kv_path = export_unified_kv_token_onnx(model, inputs, args.output_dir, opset=args.opset)
        print(f"Exported {kv_path}")
    elif args.export_unified_kv:
        print(f"Exporting unified KV-cache ONNX model to {args.output_dir}")
        kv_path = export_unified_kv_onnx(model, inputs, args.output_dir, opset=args.opset)
        print(f"Exported {kv_path}")
    elif args.export_kv:
        print(f"Exporting KV-cache ONNX models to {args.output_dir}")
        prefill_path, decode_path = export_kv_onnx(model, inputs, args.output_dir, opset=args.opset)
        print(f"Exported {prefill_path}")
        print(f"Exported {decode_path}")
    elif not args.skip_export or not onnx_path.exists():
        print(f"Exporting ONNX to {onnx_path}")
        export_onnx(model, inputs, onnx_path, opset=args.opset)
    else:
        print(f"Using existing ONNX: {onnx_path}")

    copy_runtime_files(args.model_dir, args.output_dir)
    if args.skip_validation:
        print("Skipping Top-5 validation")
        return

    session = make_ort_session(onnx_path)
    if args.validation_mode == "sample":
        print(f"Validating Top-5 logits for {args.steps} greedy steps")
        validation = validate_top5(model, session, inputs, meta, steps=args.steps)
    else:
        target_length = args.target_length or args.max_length
        print(
            "Validating Top-5 logits for every step with ONNX running target length "
            f"{target_length}"
        )
        if args.full_length_generation == "fill":
            validation = validate_full_length_top5(
                model,
                session,
                inputs,
                meta,
                target_length=target_length,
                fill_token_id=args.fill_token_id,
                chunk_size=args.compare_chunk_size,
            )
        else:
            validation = validate_full_length_greedy_top5(
                model,
                session,
                inputs,
                meta,
                target_length=target_length,
                chunk_size=args.compare_chunk_size,
                progress_interval=args.progress_interval,
            )

    report = {
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "onnx_path": str(onnx_path),
        "image": str(args.image),
        "category": args.category,
        "opset": args.opset,
        "export_config": asdict(model.config),
        "input_meta": meta,
        "validation": validation,
    }
    report_path = args.output_dir / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in validation.items() if k != "records"}
    summary["record_count"] = len(validation.get("records", []))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote validation report: {report_path}")

    if not validation["all_top5_match"]:
        raise SystemExit("Top-5 validation failed")


if __name__ == "__main__":
    main()
