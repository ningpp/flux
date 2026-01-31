import torch
from transformers import AutoModelForImageTextToText

model_path = r"D:\models\GOT-OCR-2.0-hf"
default_opset_version = 17
onnx_vision_path = r"D:\models\onnx\GOT-OCR-2.0\vision_encoder.onnx"
onnx_embed_path =  r"D:\models\onnx\GOT-OCR-2.0\embed_tokens.onnx"
onnx_lm_unified_path = r"D:\models\onnx\GOT-OCR-2.0\decoder_model.onnx"

class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector

    def forward(self, pixel_values):
        # Based on get_image_features in modeling_got_ocr2.py
        # image_outputs shape is (B, 256, 1024) if it's already flattened?
        # Let's check from our research:
        # vision_tower(pixel_values) returns GotOcr2VisionEncoderOutput
        # vision_tower(pixel_values).last_hidden_state is (B, 256, 64, 64) ?
        # Wait, neck returns (B, 256, 64, 64).
        # multi_modal_projector takes that and returns (B, 256, 1024).

        image_outputs = self.vision_tower(pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(image_outputs)
        return image_features

class LMUnified(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        # Reconstruct legacy tuple format: ((k, v), (k, v), ...)
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

        # Flatten updated past_key_values (back to legacy tuple then flat)
        legacy_pkv = outputs.past_key_values.to_legacy_cache()
        new_pkv = []
        for layer_pkv in legacy_pkv:
            new_pkv.append(layer_pkv[0])
            new_pkv.append(layer_pkv[1])

        return (logits, *new_pkv)


print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True).eval()

vision_model = VisionModelWrapper(model)

# The shape from debug_inputs.py was [1, 3, 1024, 1024]
dummy_pixel_values = torch.randn(1, 3, 1024, 1024)

print("Exporting to ONNX...")
torch.onnx.export(
    vision_model,
    (dummy_pixel_values,),
    onnx_vision_path,
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "image_features": {0: "batch_size"}
    },
    opset_version=default_opset_version
)

print(f"Done! Exported to {onnx_vision_path}")

# Qwen2's embedding layer
embed_model = model.model.language_model.embed_tokens

dummy_ids = torch.ones((1, 1), dtype=torch.long)

print("Exporting Embeddings to ONNX...")
torch.onnx.export(
    embed_model,
    (dummy_ids,),
    onnx_embed_path,
    input_names=["input_ids"],
    output_names=["inputs_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=default_opset_version
)
print(f"Done! Exported Embeddings to {onnx_embed_path}")


lm_unified = LMUnified(model)

# Dummy inputs for tracing
batch_size = 1
prefill_len = 1
num_heads = model.config.text_config.num_key_value_heads
head_dim = model.config.text_config.hidden_size // model.config.text_config.num_attention_heads
num_layers = model.config.text_config.num_hidden_layers

inputs_embeds = torch.randn(batch_size, prefill_len, 1024)
# Attention mask should be for total length = prefill_len + past_len
past_len = 1
attention_mask = torch.ones(batch_size, prefill_len + past_len, dtype=torch.long)
position_ids = torch.arange(prefill_len, dtype=torch.long).unsqueeze(0) + past_len

past_key_values = []
input_names = ["inputs_embeds", "attention_mask", "position_ids"]
output_names = ["logits"]
for i in range(num_layers):
    k = torch.randn(batch_size, num_heads, past_len, head_dim)
    v = torch.randn(batch_size, num_heads, past_len, head_dim)
    past_key_values.append(k)
    past_key_values.append(v)
    input_names.append(f"past_key_{i}")
    input_names.append(f"past_value_{i}")
    output_names.append(f"present_key_{i}")
    output_names.append(f"present_value_{i}")

print(f"Exporting Unified Language Model to ONNX...")
torch.onnx.export(
    lm_unified,
    (inputs_embeds, attention_mask, position_ids, *past_key_values),
    onnx_lm_unified_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
        **{name: {0: "batch_size", 2: "past_sequence_length"} for name in input_names[3:]},
        **{name: {0: "batch_size", 2: "total_sequence_length"} for name in output_names[1:]}
    },
    opset_version=default_opset_version,
    do_constant_folding=True
)

print(f"Done! Exported Language Model to {onnx_lm_unified_path}")
