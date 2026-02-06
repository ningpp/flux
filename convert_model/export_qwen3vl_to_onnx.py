"""
Export Qwen3-VL-2B-Instruct to ONNX format.
Run: D:\\conda\\envs\\qwen3vlonnx\\python convert_model\\export_qwen3vl_to_onnx.py

Exports 3 ONNX files:
  1. vision_encoder.onnx    — Qwen3VLVisionModel (pixel_values + image_grid_thw → image_features + deepstack_0/1/2)
  2. embed_tokens.onnx      — Embedding layer (input_ids → inputs_embeds)
  3. decoder_model_merged.onnx — Text decoder with KV-cache + deepstack fusion (inputs_embeds + position_ids + ... → logits + present KV)

Architecture: 28 decoder layers, 16 heads (8 KV heads, GQA), hidden=2048, head_dim=128, vocab=151936
DeepStack: layers 5,11,17 → 3 extra vision feature tensors injected into decoder layers 0,1,2
MRoPE: 3D position_ids [3, batch, seq] with interleaved sections [24,20,20]
Weight tying: lm_head.weight == embed_tokens.weight
"""
import os
import gc
import torch
import torch.nn as nn
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
    """Export vision encoder: pixel_values + image_grid_thw → image_features + deepstack_0/1/2"""
    print("\n=== Exporting Vision Encoder ===")

    class VisionEncoderWrapper(nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def forward(self, hidden_states, grid_thw):
            output = self.visual(hidden_states, grid_thw=grid_thw, return_dict=True)
            image_features = output.pooler_output  # merged features [num_merged, 2048]
            ds0 = output.deepstack_features[0]  # [num_merged, 2048]
            ds1 = output.deepstack_features[1]
            ds2 = output.deepstack_features[2]
            return image_features, ds0, ds1, ds2

    wrapper = VisionEncoderWrapper(model.model.visual).eval()

    # Dummy inputs: 1 image, grid_thw = [1, 12, 24] → 288 patches
    # pixel_values shape: [num_patches, 1536] where 1536 = 3*2*16*16
    num_patches = 288  # 1 * 12 * 24
    dummy_pixel_values = torch.randn(num_patches, 1536, dtype=torch.float32)
    dummy_grid_thw = torch.tensor([[1, 12, 24]], dtype=torch.long)

    # Test forward
    with torch.no_grad():
        out = wrapper(dummy_pixel_values, dummy_grid_thw)
        print(f"  image_features shape: {out[0].shape}")
        print(f"  deepstack_0 shape: {out[1].shape}")
        print(f"  deepstack_1 shape: {out[2].shape}")
        print(f"  deepstack_2 shape: {out[3].shape}")

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
    print(f"  Saved: {onnx_path}")
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {file_size:.1f} MB")


def export_embed_tokens(model):
    """Export embedding layer: input_ids → inputs_embeds"""
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
    print(f"  Saved: {onnx_path}")
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {file_size:.1f} MB")


def export_decoder(model):
    """Export decoder with KV-cache + deepstack fusion."""
    print("\n=== Exporting Decoder (merged prefill/decode with KV-cache + deepstack) ===")

    class DecoderWrapper(nn.Module):
        """Wraps Qwen3VLTextModel + lm_head with explicit KV-cache I/O and deepstack.

        Deepstack is ALWAYS applied (no conditional branching).
        For decode steps (no visual tokens), pass:
          - visual_pos_mask = all-False [batch, seq]
          - deepstack_features_* = empty [0, hidden_size]
        This makes the boolean indexing a no-op (selects 0 elements).
        """
        def __init__(self, language_model, lm_head):
            super().__init__()
            self.language_model = language_model
            self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
            self.lm_head.weight = lm_head.weight

        def forward(self, inputs_embeds, attention_mask, position_ids,
                    deepstack_features_0, deepstack_features_1, deepstack_features_2,
                    visual_pos_mask,
                    *past_key_values_flat):
            """
            Args:
                inputs_embeds: [batch, seq_len, 2048]
                attention_mask: [batch, total_len]
                position_ids: [3, batch, seq_len] — T/H/W MRoPE positions
                deepstack_features_0/1/2: [num_vis_tokens, 2048] — deepstack per layer
                visual_pos_mask: [batch, seq_len] — bool mask (True = visual token position)
                *past_key_values_flat: alternating key, value tensors for each layer
                    each [batch, num_kv_heads, past_len, head_dim]
            Returns:
                logits: [batch, seq_len, vocab_size]
                present key/value tensors for each layer
            """
            # Reconstruct DynamicCache from flat KV tensors
            past_key_values = DynamicCache()
            for i in range(NUM_LAYERS):
                past_key_values.update(
                    past_key_values_flat[i * 2],      # key
                    past_key_values_flat[i * 2 + 1],   # value
                    layer_idx=i,
                )

            # Always pass deepstack — no conditional branching
            visual_pos_masks_bool = visual_pos_mask.bool()
            deepstack_visual_embeds = [
                deepstack_features_0,
                deepstack_features_1,
                deepstack_features_2,
            ]

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                visual_pos_masks=visual_pos_masks_bool,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

            logits = self.lm_head(outputs.last_hidden_state)

            # Extract KV cache as flat tensors via .layers accessor
            cache = outputs.past_key_values
            new_pkv = []
            for i in range(NUM_LAYERS):
                new_pkv.append(cache.layers[i].keys)    # [batch, num_kv_heads, total_len, head_dim]
                new_pkv.append(cache.layers[i].values)

            return (logits, *new_pkv)

    wrapper = DecoderWrapper(model.model.language_model, model.lm_head).eval()

    # Dummy inputs for tracing — include visual tokens so deepstack path is traced
    batch_size = 1
    seq_len = 5       # a few tokens including visual tokens
    past_len = 0      # empty past for tracing (will init with zero-length KV)
    num_vis_tokens = 2  # 2 visual token positions in the sequence

    dummy_inputs_embeds = torch.randn(batch_size, seq_len, HIDDEN_SIZE)
    dummy_attention_mask = torch.ones(batch_size, seq_len + past_len, dtype=torch.long)
    dummy_position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long)

    # Deepstack features matching number of visual tokens
    dummy_ds0 = torch.randn(num_vis_tokens, HIDDEN_SIZE)
    dummy_ds1 = torch.randn(num_vis_tokens, HIDDEN_SIZE)
    dummy_ds2 = torch.randn(num_vis_tokens, HIDDEN_SIZE)

    # Visual position mask — positions 1 and 3 are visual tokens
    dummy_visual_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    dummy_visual_mask[0, 1] = True
    dummy_visual_mask[0, 3] = True

    past_key_values = []
    input_names = [
        "inputs_embeds", "attention_mask", "position_ids",
        "deepstack_features_0", "deepstack_features_1", "deepstack_features_2",
        "visual_pos_mask",
    ]
    output_names = ["logits"]

    for i in range(NUM_LAYERS):
        # Empty past KV for prefill tracing
        k = torch.zeros(batch_size, NUM_KV_HEADS, past_len, HEAD_DIM)
        v = torch.zeros(batch_size, NUM_KV_HEADS, past_len, HEAD_DIM)
        past_key_values.append(k)
        past_key_values.append(v)
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    args = (dummy_inputs_embeds, dummy_attention_mask, dummy_position_ids,
            dummy_ds0, dummy_ds1, dummy_ds2, dummy_visual_mask,
            *past_key_values)

    # Test forward
    with torch.no_grad():
        out = wrapper(*args)
        print(f"  logits shape: {out[0].shape}")
        print(f"  present.0.key shape: {out[1].shape}")

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {1: "batch_size", 2: "sequence_length"},
        "deepstack_features_0": {0: "num_visual_tokens"},
        "deepstack_features_1": {0: "num_visual_tokens"},
        "deepstack_features_2": {0: "num_visual_tokens"},
        "visual_pos_mask": {0: "batch_size", 1: "sequence_length"},
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
    print(f"  Saved: {onnx_path}")
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {file_size:.1f} MB")


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
    print("Qwen3-VL-2B-Instruct → ONNX Export")
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

    # 3. Decoder with KV-cache + deepstack
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
