"""
LightOnOCR-2-1B Torch Inference (reference implementation).
Run: D:\\conda\\envs\\qwen3vlonnx\\python scripts\\lightonocr_torch_infer.py

Uses transformers >= 5.0 LightOnOcrForConditionalGeneration.
Greedy decoding (do_sample=False) for deterministic output.
"""
import time
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image

MODEL_DIR = r"D:\models\LightOnOCR-2-1B"
IMAGE_PATH = r"D:\tmp\formula_2025-8-2_17-28-16.jpg"

def main():
    print("=" * 60)
    print("LightOnOCR-2-1B  —  Torch Inference (greedy)")
    print("=" * 60)

    # Load model + processor
    t0 = time.time()
    processor = LightOnOcrProcessor.from_pretrained(MODEL_DIR)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Build conversation
    conversation = [
        {"role": "user", "content": [{"type": "image", "url": IMAGE_PATH}]}
    ]

    # Apply chat template  →  tokenize + prepare pixel values
    t1 = time.time()
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prep_time = time.time() - t1
    print(f"Preprocessing: {prep_time:.3f}s")

    # Debug: print input shapes & image token count
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    print(f"input_ids shape: {input_ids.shape}")
    print(f"pixel_values shape: {pixel_values.shape}")

    image_token_id = 151655
    num_image_tokens = (input_ids == image_token_id).sum().item()
    print(f"Number of <|image_pad|> tokens: {num_image_tokens}")
    print(f"input_ids[:20] = {input_ids[0, :20].tolist()}")

    # Find all unique special tokens in the input
    unique_ids = input_ids[0].unique().tolist()
    special = [t for t in unique_ids if t >= 151643]
    print(f"Special token IDs in input: {special}")

    # Generate with greedy decoding
    t2 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
    gen_time = time.time() - t2
    new_ids = output_ids[0, input_ids.shape[1]:]
    text = processor.decode(new_ids, skip_special_tokens=True)

    print(f"\nGeneration: {gen_time:.2f}s  ({len(new_ids)} tokens)")
    print(f"First 10 generated token IDs: {new_ids[:10].tolist()}")
    print(f"\n--- GENERATED TEXT ---")
    print(text)
    print(f"--- END ---")

if __name__ == "__main__":
    main()
