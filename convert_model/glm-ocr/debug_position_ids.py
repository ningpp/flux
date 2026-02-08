"""
Debug: Compare position_ids used by model.generate() vs manual decode loop.
The ONNX models are numerically correct - the issue is in position_ids computation.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

MODEL_PATH = r"D:\models\GLM-OCR"
IMAGE_PATH = r"D:\tmp\formula-2026028-105537.jpg"
PROMPT = "Formula Recognition:"

image = Image.open(IMAGE_PATH).convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

# Prepare inputs
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
position_ids = inputs.get("position_ids", None)
seq_len = input_ids.shape[1]

print(f"input_ids shape: {input_ids.shape}")
print(f"position_ids: {position_ids is not None}")
if position_ids is not None:
    print(f"position_ids shape: {position_ids.shape}")
    print(f"position_ids[:, :, -5:] =")
    print(position_ids[:, :, -5:])
else:
    print("position_ids is None - model computes internally")

# Load model
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="float32")
model.eval()

# ============================================================
# Hook into model to capture position_ids at each step
# ============================================================
captured_positions = []
captured_cache_len = []
captured_attn_mask_len = []

# Hook the language_model's forward to see position_ids
original_lm_forward = model.model.language_model.forward

def hooked_lm_forward(*args, **kwargs):
    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
        pos = kwargs["position_ids"].clone().detach()
        captured_positions.append(pos)
    elif len(args) > 2 and args[2] is not None:  # position_ids is 3rd arg
        captured_positions.append(args[2].clone().detach())
    else:
        captured_positions.append(None)
    
    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        captured_attn_mask_len.append(kwargs["attention_mask"].shape[-1])
    else:
        captured_attn_mask_len.append(-1)
    
    if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
        try:
            pkv = kwargs["past_key_values"]
            if hasattr(pkv, 'get_seq_length'):
                captured_cache_len.append(pkv.get_seq_length())
            elif isinstance(pkv, (tuple, list)) and len(pkv) > 0:
                captured_cache_len.append(pkv[0][0].shape[2])
            else:
                captured_cache_len.append(-1)
        except:
            captured_cache_len.append(-1)
    else:
        captured_cache_len.append(0)
    
    return original_lm_forward(*args, **kwargs)

model.model.language_model.forward = hooked_lm_forward

# Run generate
print("\nRunning model.generate()...")
gen_inputs = {
    "input_ids": input_ids,
    "pixel_values": inputs["pixel_values"],
    "image_grid_thw": inputs["image_grid_thw"],
    "attention_mask": torch.ones_like(input_ids),
    "max_new_tokens": 60,
}

with torch.no_grad():
    gen_output = model.generate(**gen_inputs)

gen_tokens = gen_output[0][seq_len:].tolist()
print(f"Generated {len(gen_tokens)} tokens")
print(f"Captured {len(captured_positions)} position_ids snapshots")

# ============================================================
# Print captured position_ids from generate()
# ============================================================
print("\n===== CAPTURED POSITION_IDS FROM model.generate() =====")
print(f"Captured {len(captured_positions)} forward calls")

for step in range(min(60, len(captured_positions))):
    cap_pos = captured_positions[step]
    cache_len = captured_cache_len[step] if step < len(captured_cache_len) else "?"
    attn_len = captured_attn_mask_len[step] if step < len(captured_attn_mask_len) else "?"
    
    if cap_pos is None:
        print(f"{step:>4} | pos_ids=None | cache_len={cache_len} | attn_mask_len={attn_len}")
    elif cap_pos.dim() == 3 and cap_pos.shape[-1] <= 3:
        # Decode step - small position_ids
        print(f"{step:>4} | shape={list(cap_pos.shape)} vals={cap_pos.squeeze().tolist()} | cache_len={cache_len} | attn_mask_len={attn_len}")
    elif cap_pos.dim() == 3:
        # Prefill - show last 5
        print(f"{step:>4} | shape={list(cap_pos.shape)} last5={cap_pos[:,:,-5:].squeeze().tolist()} | cache_len={cache_len} | attn_mask_len={attn_len}")
    else:
        print(f"{step:>4} | shape={list(cap_pos.shape)} | cache_len={cache_len} | attn_mask_len={attn_len}")

# Print generated text
print(f"\ngenerate() text:")
print(tokenizer.decode(gen_tokens, skip_special_tokens=True))
print(f"\ngenerate() tokens: {gen_tokens}")
