"""
Create a mixed Float16 GOT-OCR-2.0 ONNX model.

All three sub-models are FP16 to maximize memory savings and GPU throughput:
- vision_encoder.onnx   exported directly from PyTorch in FP16
- embed_tokens.onnx     exported directly from PyTorch in FP16
- decoder_model.onnx    converted from the existing FP32 decoder ONNX using
                        onnxruntime.transformers.float16 (keep_io_types=False)

External I/O of every model is FP16. Downstream callers must feed FP16 tensors
(or wrap inputs/outputs with Cast nodes for FP32 callers).

Run with train_demo conda environment:
    conda run -n train_demo python convert_model\convert_gotocr2_to_fp16.py
"""
import os
import shutil
import json
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from transformers import AutoModelForImageTextToText
from onnxruntime.transformers.float16 import convert_float_to_float16

SRC_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0")
DST_DIR = Path(r"D:\models\onnx\GOT-OCR-2.0-FP16")
MODEL_PATH = Path(r"D:\models\GOT-OCR-2.0-hf")
OPSET_VERSION = 17


class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector

    def forward(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(image_outputs)
        return image_features


class LMUnified(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        pkv_legacy = []
        for i in range(0, len(past_key_values), 2):
            pkv_legacy.append((past_key_values[i], past_key_values[i + 1]))

        from transformers.cache_utils import DynamicCache
        pkv = DynamicCache.from_legacy_cache(tuple(pkv_legacy))

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True
        )
        logits = self.lm_head(outputs.last_hidden_state)

        legacy_pkv = outputs.past_key_values.to_legacy_cache()
        new_pkv = []
        for layer_pkv in legacy_pkv:
            new_pkv.append(layer_pkv[0])
            new_pkv.append(layer_pkv[1])

        return (logits, *new_pkv)


def copy_non_model_files(src: Path, dst: Path):
    """Copy tokenizer / config files that are not the three large ONNX models."""
    for item in src.iterdir():
        if item.name in {
            "vision_encoder.onnx",
            "decoder_model.onnx",
            "embed_tokens.onnx",
        }:
            continue
        dst_item = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)
        print(f"  copied {item.name}")


def export_vision_encoder_fp16(model, dst_path: Path):
    print(f"\nExporting vision encoder to FP16: {dst_path}")
    vision_model = VisionModelWrapper(model).half().eval()
    dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float16)
    torch.onnx.export(
        vision_model,
        (dummy,),
        str(dst_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"}
        },
        opset_version=OPSET_VERSION,
    )


def export_embed_tokens_fp16(model, dst_path: Path):
    print(f"\nExporting embed_tokens to FP16: {dst_path}")
    embed = model.model.language_model.embed_tokens.half().eval()
    dummy = torch.ones((1, 1), dtype=torch.long)
    torch.onnx.export(
        embed,
        (dummy,),
        str(dst_path),
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=OPSET_VERSION,
    )


def convert_decoder_fp16(src_path: Path, dst_path: Path):
    print(f"\nConverting decoder to FP16: {dst_path}")
    model = onnx.load(str(src_path))
    fp16 = convert_float_to_float16(
        model,
        keep_io_types=False,
        min_positive_val=5.96e-08,
        max_finite_val=65504.0,
    )
    onnx.save(fp16, str(dst_path))


def verify_loads(model_dir: Path):
    print("\nVerifying FP16 models can be loaded...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CPUExecutionProvider"]
    for name in ("vision_encoder.onnx", "embed_tokens.onnx", "decoder_model.onnx"):
        session = ort.InferenceSession(str(model_dir / name), sess_options, providers=providers)
        print(f"  {name}: OK (inputs={[i.name for i in session.get_inputs()]}, outputs={[o.name for o in session.get_outputs()]})")


def update_config_json(dst_dir: Path):
    marker_path = dst_dir / "fp16_config.json"
    cfg = {
        "precision": "mixed_float16",
        "vision_encoder": "float16",
        "decoder_model": "float16",
        "embed_tokens": "float16",
        "io_dtype": "float16",
        "source": str(SRC_DIR),
    }
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote {marker_path.name}")


def main():
    print("=" * 60)
    print("GOT-OCR-2.0 -> Mixed FP16")
    print("=" * 60)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for p in (SRC_DIR / "decoder_model.onnx",):
        if not p.exists():
            raise FileNotFoundError(f"Missing source model: {p}")

    print(f"Loading PyTorch model from {MODEL_PATH} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, torch_dtype=torch.float16
    ).eval().half()

    export_vision_encoder_fp16(model, DST_DIR / "vision_encoder.onnx")
    export_embed_tokens_fp16(model, DST_DIR / "embed_tokens.onnx")
    convert_decoder_fp16(SRC_DIR / "decoder_model.onnx", DST_DIR / "decoder_model.onnx")

    print("\nCopying tokenizer/config files...")
    copy_non_model_files(SRC_DIR, DST_DIR)
    update_config_json(DST_DIR)

    verify_loads(DST_DIR)

    print("\n" + "=" * 60)
    print("Size summary")
    print("=" * 60)
    for name in ("vision_encoder.onnx", "decoder_model.onnx", "embed_tokens.onnx"):
        src_size = (SRC_DIR / name).stat().st_size / (1024 * 1024)
        dst_size = (DST_DIR / name).stat().st_size / (1024 * 1024)
        print(f"  {name}: {src_size:.1f} MB -> {dst_size:.1f} MB")

    print("\nDone. FP16 model is ready at:", DST_DIR)


if __name__ == "__main__":
    main()
