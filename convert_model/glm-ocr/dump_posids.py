"""Dump full position_ids from model.generate() prefill to understand the pattern."""
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

MODEL_PATH = r"D:\models\GLM-OCR"
IMAGE_PATH = r"D:\tmp\formula-2026028-105537.jpg"
PROMPT = "Formula Recognition:"

image = Image.open(IMAGE_PATH).convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
image_grid_thw = inputs["image_grid_thw"]

print(f"input_ids shape: {input_ids.shape}")
print(f"image_grid_thw: {image_grid_thw}")
print(f"image_token_id: {config.image_token_id}")

# Find image token positions
img_mask = (input_ids[0] == config.image_token_id)
img_positions = torch.where(img_mask)[0]
print(f"Image tokens: {img_mask.sum().item()} at positions [{img_positions[0].item()}..{img_positions[-1].item()}]")
print(f"Text tokens before image: {img_positions[0].item()}")
print(f"Text tokens after image: {input_ids.shape[1] - img_positions[-1].item() - 1}")

# Hook to capture position_ids
captured_pos = [None]
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="float32")
model.eval()

orig_fwd = model.model.language_model.forward
def hook(*args, **kwargs):
    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
        captured_pos[0] = kwargs["position_ids"].clone().detach()
    return orig_fwd(*args, **kwargs)
model.model.language_model.forward = hook

with torch.no_grad():
    gen_out = model.generate(
        input_ids=input_ids,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=image_grid_thw,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=1,
    )

pos = captured_pos[0]  # [3, 1, 237]
print(f"\nposition_ids shape: {pos.shape}")
print(f"\nDim 0 (temporal):")
print(pos[0, 0, :].tolist())
print(f"\nDim 1 (height):")
print(pos[1, 0, :].tolist())
print(f"\nDim 2 (width):")
print(pos[2, 0, :].tolist())

# Analyze vision token positions
vstart = img_positions[0].item()
vend = img_positions[-1].item() + 1
print(f"\n--- Vision token positions (indices {vstart}..{vend-1}) ---")
print(f"Dim 0 vision: {pos[0, 0, vstart:vend].tolist()}")
print(f"Dim 1 vision: {pos[1, 0, vstart:vend].tolist()}")
print(f"Dim 2 vision: {pos[2, 0, vstart:vend].tolist()}")

print(f"\n--- Text before vision (indices 0..{vstart-1}) ---")
print(f"Dim 0: {pos[0, 0, :vstart].tolist()}")
print(f"Dim 1: {pos[1, 0, :vstart].tolist()}")
print(f"Dim 2: {pos[2, 0, :vstart].tolist()}")

print(f"\n--- Text after vision (indices {vend}..{input_ids.shape[1]-1}) ---")
print(f"Dim 0: {pos[0, 0, vend:].tolist()}")
print(f"Dim 1: {pos[1, 0, vend:].tolist()}")
print(f"Dim 2: {pos[2, 0, vend:].tolist()}")
