import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnxruntime as ort
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = r"D:\models\blip-image-captioning-base"
ONNX_DIR = r"D:\models\onnx\blip-image-captioning-base"
IMAGE_PATHS = [
    r"D:\tmp\img-2026-02-07-120018.png",
    r"D:\tmp\img-2026-02-07-120114.png"
]

ENCODER_ONNX_PATH = os.path.join(ONNX_DIR, "blip_vision_encoder.onnx")
DECODER_ONNX_PATH = os.path.join(ONNX_DIR, "blip_text_decoder.onnx")

def run_validation():
    print(f"Loading processor from {MODEL_PATH}...")
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    pytorch_model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    pytorch_model.eval()

    print("\n--- Running Validation ---")
    
    # Load ONNX sessions
    if not os.path.exists(ENCODER_ONNX_PATH) or not os.path.exists(DECODER_ONNX_PATH):
        print("ONNX models not found. Please run conversion first.")
        return

    try:
        print(f"Loading ONNX Encoder: {ENCODER_ONNX_PATH}")
        enc_session = ort.InferenceSession(ENCODER_ONNX_PATH, providers=['CPUExecutionProvider'])
        print(f"Loading ONNX Decoder: {DECODER_ONNX_PATH}")
        dec_session = ort.InferenceSession(DECODER_ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX models: {e}")
        return

    # Process Images
    raw_images = []
    print(f"Loading images: {IMAGE_PATHS}")
    valid_paths = []
    for p in IMAGE_PATHS:
        if os.path.exists(p):
            try:
                raw_images.append(Image.open(p).convert('RGB'))
                valid_paths.append(p)
            except Exception as e:
                 print(f"Error loading image {p}: {e}")
        else:
            print(f"Warning: Image not found {p}")

    if not raw_images:
        print("No valid images found for validation.")
        return

    inputs = processor(images=raw_images, return_tensors="pt")
    
    # 1. PyTorch Baseline
    print("Generating captions with PyTorch (Original)...")
    with torch.no_grad():
        # Use max_new_tokens explicitly to align with the loop count
        pt_out = pytorch_model.generate(**inputs, max_new_tokens=20)
        pt_captions = processor.batch_decode(pt_out, skip_special_tokens=True)

    # 2. ONNX Inference (Greedy)
    print("Generating captions with ONNX (Converted)...")
    onnx_captions = []
    pixel_values_np = inputs.pixel_values.numpy()

    # Get special tokens
    bos_token_id = pytorch_model.config.text_config.bos_token_id
    if bos_token_id is None: bos_token_id = 30522
    sep_token_id = pytorch_model.config.text_config.sep_token_id
    if sep_token_id is None: sep_token_id = 102

    for i in range(len(raw_images)):
        # -- Vision Encoder --
        img_input = pixel_values_np[i:i+1] # Batch size 1
        encoder_inputs = {'pixel_values': img_input}
        encoder_hidden_states = enc_session.run(None, encoder_inputs)[0]
        
        # -- Text Decoder Loop --
        cur_input_ids = np.array([[bos_token_id]], dtype=np.int64)
        
        for _ in range(20): # Max length
            decoder_inputs = {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
                # attention_mask is pruned
            }
            logits = dec_session.run(None, decoder_inputs)[0]
            
            # Greedy: take argmax of the last token
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)[0]
            
            if next_token_id == sep_token_id:
                break
            
            cur_input_ids = np.concatenate([cur_input_ids, [[next_token_id]]], axis=1)
        
        # Decode tokens
        caption = processor.decode(cur_input_ids[0], skip_special_tokens=True)
        onnx_captions.append(caption)

    # Validate
    print("\n--- Results Comparison ---")
    match_count = 0
    for i, (pt, onnx) in enumerate(zip(pt_captions, onnx_captions)):
        print(f"\nImage {i+1} ({valid_paths[i]}):")
        print(f"  PyTorch: {pt}")
        print(f"  ONNX:    {onnx}")
        if pt == onnx:
            print("  Status:  MATCH ✅")
            match_count += 1
        else:
            print("  Status:  MISMATCH ❌")
            
    if match_count == len(pt_captions):
        print("\nAll validations passed!")
    else:
        print(f"\nValidation failed for {len(pt_captions) - match_count} images.")

if __name__ == "__main__":
    run_validation()
