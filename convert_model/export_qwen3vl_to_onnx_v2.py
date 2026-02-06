"""
Export Qwen3-VL-2B-Instruct to ONNX format.
Run: D:\\conda\\envs\\qwen3vlonnx\\python convert_model\\export_qwen3vl_to_onnx.py

Exports 3 ONNX files:
  1. vision_encoder.onnx    - Qwen3VLVisionModel (pixel_values + image_grid_thw -> image_features + deepstack_0/1/2)
  2. embed_tokens.onnx      - Embedding layer (input_ids -> inputs_embeds)
  3. decoder_model_merged.onnx - Text decoder with KV-cache + deepstack fusion

Architecture: 28 decoder layers, 16 heads (8 KV heads, GQA), hidden=2048, head_dim=128, vocab=151936
DeepStack: layers 5,11,17 -> 3 extra vision feature tensors injected into decoder layers 0,1,2
MRoPE: 3D position_ids [3, batch, seq] with interleaved sections [24,20,20]
Weight tying: lm_head.weight == embed_tokens.weight

Key design: Deepstack uses PRE-SCATTERED tensors [batch, seq, hidden] to avoid
boolean indexing (which doesn't export dynamically via torch.onnx.export).
The caller scatters vision features into the right positions before calling decoder.
"""
import os
import gc
import torch
import torch.nn as nn
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

MODEL_DIR = r"D:\models\Qwen3-VL-2B-Instruct"
ONNX_DIR = r"D:\models\onnx\Qwen3-VL-2B-Instruct"
OPSET_VERSION = 17

# Model constants
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936


def export_vision_encoder(model):
    """Export vision encoder: pixel_values + image_grid_thw -> image_features + deepstack_0/1/2"""
    print("\n=== Exporting Vision Encoder ===")

    class VisionEncoderWrapper(nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def forward(self, hidden_states, grid_thw):
            output = self.visual(hidden_states, grid_thw=grid_thw, return_dict=True)
            image_features = output.pooler_output
            ds0 = output.deepstack_features[0]
            ds1 = output.deepstack_features[1]
            ds2 = output.deepstack_features[2]
            return image_features, ds0, ds1, ds2

    wrapper = VisionEncoderWrapper(model.model.visual).eval()

    # Dummy: 1 image, grid_thw = [1, 12, 24] -> 288 patches
    num_patches = 288
    dummy_pixel_values = torch.randn(num_patches, 1536, dtype=torch.float32)
    dummy_grid_thw = torch.tensor([[1, 12, 24]], dtype=torch.long)

    with torch.no_grad():
        out = wrapper(dummy_pixel_values, dummy_grid_thw)
        print(f"  image_features shape: {out[0].shape}")
        print(f"  deepstack_0 shape: {out[1].shape}")

    onnx_path = os.path.join(ONNX_DIR, "vision_encoder.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_grid_thw),
        onnx_path,
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["image_features", "deepstack_features_0", "deepstack_features_1", "deepstack_features_2"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
            "image_features": {0: "num_merged_patches"},
            "deepstack_features_0": {0: "num_merged_patches"},
            "deepstack_features_1": {0: "num_merged_patches"},
            "deepstack_features_2": {0: "num_merged_patches"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")


def export_embed_tokens(model):
    """Export embedding layer: input_ids -> inputs_embeds"""
    print("\n=== Exporting Embed Tokens ===")

    embed_model = model.model.language_model.embed_tokens
    dummy_ids = torch.ones((1, 1), dtype=torch.long)
    onnx_path = os.path.join(ONNX_DIR, "embed_tokens.onnx")

    torch.onnx.export(
        embed_model,
        (dummy_ids,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=OPSET_VERSION,
    )
    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")


def export_decoder(model):
    """Export decoder with KV-cache + pre-scattered deepstack fusion.

    Deepstack tensors are passed as [batch, seq_len, hidden_size] with values
    already placed at the correct visual token positions (zeros elsewhere).
    This avoids boolean indexing which doesn't trace dynamically.
    """
    print("\n=== Exporting Decoder (pre-scattered deepstack) ===")

    # Monkey-patch _deepstack_process to use simple addition instead of boolean indexing
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    original_deepstack = Qwen3VLTextModel._deepstack_process

    def _deepstack_additive(self, hidden_states, visual_pos_masks, visual_embeds):
        """Pre-scattered: visual_embeds is [batch, seq, hidden], just add."""
        return hidden_states + visual_embeds

    Qwen3VLTextModel._deepstack_process = _deepstack_additive

    class DecoderWrapper(nn.Module):
        def __init__(self, language_model, lm_head):
            super().__init__()
            self.language_model = language_model
            self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
            self.lm_head.weight = lm_head.weight

        def forward(self, inputs_embeds, attention_mask, position_ids,
                    deepstack_scattered_0, deepstack_scattered_1, deepstack_scattered_2,
                    *past_key_values_flat):
            """
            Args:
                inputs_embeds: [batch, seq_len, 2048]
                attention_mask: [batch, total_len]
                position_ids: [3, batch, seq_len]
                deepstack_scattered_0/1/2: [batch, seq_len, 2048] - pre-scattered
                *past_key_values_flat: 56 tensors (28 layers x key/value)
            Returns:
                logits + 56 present KV tensors
            """
            # Build DynamicCache
            past_key_values = DynamicCache()
            for i in range(NUM_LAYERS):
                past_key_values.update(
                    past_key_values_flat[i * 2],
                    past_key_values_flat[i * 2 + 1],
                    layer_idx=i,
                )

            # Pre-scattered deepstack: pass as list, visual_pos_masks is a dummy (unused)
            # _deepstack_process is monkey-patched to just add
            dummy_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1],
                                    dtype=torch.bool, device=inputs_embeds.device)
            deepstack_visual_embeds = [
                deepstack_scattered_0,
                deepstack_scattered_1,
                deepstack_scattered_2,
            ]

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                visual_pos_masks=dummy_mask,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            logits = self.lm_head(outputs.last_hidden_state)

            cache = outputs.past_key_values
            new_pkv = []
            for i in range(NUM_LAYERS):
                new_pkv.append(cache.layers[i].keys)
                new_pkv.append(cache.layers[i].values)

            return (logits, *new_pkv)

    wrapper = DecoderWrapper(model.model.language_model, model.lm_head).eval()

    batch_size = 1
    seq_len = 5
    past_len = 0

    dummy_inputs_embeds = torch.randn(batch_size, seq_len, HIDDEN_SIZE)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long)

    # Pre-scattered deepstack: [batch, seq_len, hidden]
    dummy_ds0 = torch.zeros(batch_size, seq_len, HIDDEN_SIZE)
    dummy_ds1 = torch.zeros(batch_size, seq_len, HIDDEN_SIZE)
    dummy_ds2 = torch.zeros(batch_size, seq_len, HIDDEN_SIZE)

    past_key_values = []
    input_names = [
        "inputs_embeds", "attention_mask", "position_ids",
        "deepstack_scattered_0", "deepstack_scattered_1", "deepstack_scattered_2",
    ]
    output_names = ["logits"]

    for i in range(NUM_LAYERS):
        k = torch.zeros(batch_size, NUM_KV_HEADS, past_len, HEAD_DIM)
        v = torch.zeros(batch_size, NUM_KV_HEADS, past_len, HEAD_DIM)
        past_key_values.append(k)
        past_key_values.append(v)
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    args = (dummy_inputs_embeds, dummy_attention_mask, dummy_position_ids,
            dummy_ds0, dummy_ds1, dummy_ds2,
            *past_key_values)

    with torch.no_grad():
        out = wrapper(*args)
        print(f"  logits shape: {out[0].shape}")
        print(f"  present.0.key shape: {out[1].shape}")

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {1: "batch_size", 2: "sequence_length"},
        "deepstack_scattered_0": {0: "batch_size", 1: "sequence_length"},
        "deepstack_scattered_1": {0: "batch_size", 1: "sequence_length"},
        "deepstack_scattered_2": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for i in range(NUM_LAYERS):
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_sequence_length"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_sequence_length"}
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "total_sequence_length"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "total_sequence_length"}

    onnx_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx")
    print("  Exporting... (this may take a few minutes)")

    torch.onnx.export(
        wrapper,
        args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Consolidate all external data files into a single .onnx.data file
    print("  Consolidating external data into single file...")
    model_proto = onnx.load(onnx_path, load_external_data=True)
    # Remove old scattered external data files
    import glob
    for f in glob.glob(os.path.join(ONNX_DIR, "onnx__MatMul_*")):
        os.remove(f)
    for f in glob.glob(os.path.join(ONNX_DIR, "language_model.layers.*")):
        os.remove(f)
    for f in glob.glob(os.path.join(ONNX_DIR, "language_model.norm.*")):
        os.remove(f)
    # Save with all weights in one external file
    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location="decoder_model_merged.onnx.data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(model_proto, onnx_path)
    data_path = os.path.join(ONNX_DIR, "decoder_model_merged.onnx.data")
    print(f"  Consolidated: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")
    print(f"  External data: {data_path} ({os.path.getsize(data_path)/1024/1024:.1f} MB)")

    # Restore original
    Qwen3VLTextModel._deepstack_process = original_deepstack


def copy_tokenizer_files():
    """Copy tokenizer files to ONNX model directory."""
    import shutil
    print("\n=== Copying Tokenizer Files ===")
    for fname in ["tokenizer.json", "tokenizer_config.json", "config.json",
                   "generation_config.json", "preprocessor_config.json"]:
        src = os.path.join(MODEL_DIR, fname)
        dst = os.path.join(ONNX_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")


def main():
    print("=" * 60)
    print("Qwen3-VL-2B-Instruct -> ONNX Export")
    print("=" * 60)

    os.makedirs(ONNX_DIR, exist_ok=True)

    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    print("Model loaded.")

    # 1. Vision encoder
    vision_path = os.path.join(ONNX_DIR, "vision_encoder.onnx")
    if os.path.exists(vision_path):
        print(f"\n  Skipping vision encoder (already exists: {vision_path})")
    else:
        export_vision_encoder(model)
    gc.collect()

    # 2. Embed tokens
    embed_path = os.path.join(ONNX_DIR, "embed_tokens.onnx")
    if os.path.exists(embed_path):
        print(f"\n  Skipping embed tokens (already exists: {embed_path})")
    else:
        export_embed_tokens(model)
    gc.collect()

    # 3. Decoder with pre-scattered deepstack
    export_decoder(model)
    gc.collect()

    # 4. Copy tokenizer files
    copy_tokenizer_files()

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"ONNX models saved to: {ONNX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
