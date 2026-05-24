import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnxruntime as ort
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = r"D:\models\blip-image-captioning-large"
ONNX_DIR = r"D:\models\onnx\blip-image-captioning-large"
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
    
    if not os.path.exists(ENCODER_ONNX_PATH) or not os.path.exists(DECODER_ONNX_PATH):
        print("ONNX models not found.")
        return

    try:
        print(f"Loading ONNX Encoder: {ENCODER_ONNX_PATH}")
        enc_session = ort.InferenceSession(ENCODER_ONNX_PATH, providers=['CPUExecutionProvider'])
        print(f"Loading ONNX Decoder: {DECODER_ONNX_PATH}")
        dec_session = ort.InferenceSession(DECODER_ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX models: {e}")
        return

    raw_images = []
    for p in IMAGE_PATHS:
        if os.path.exists(p):
            raw_images.append(Image.open(p).convert('RGB'))

    if not raw_images:
        print("No valid images found.")
        return

    inputs = processor(images=raw_images, return_tensors="pt")
    
    # 1. PyTorch Baseline
    print("Generating captions with PyTorch...", flush=True)
    with torch.no_grad():
        pt_out = pytorch_model.generate(**inputs, max_new_tokens=20)
        pt_captions = processor.batch_decode(pt_out, skip_special_tokens=True)

    # 2. ONNX Inference
    print("Generating captions with ONNX...", flush=True)
    onnx_captions = []
    pixel_values_np = inputs.pixel_values.numpy()

    bos_token_id = pytorch_model.config.text_config.bos_token_id or 30522
    sep_token_id = pytorch_model.config.text_config.sep_token_id or 102

    for i in range(len(raw_images)):
        img_input = pixel_values_np[i:i+1]
        
        # Run Encoder
        encoder_inputs = {'pixel_values': img_input}
        encoder_hidden_states = enc_session.run(None, encoder_inputs)[0]
        
        # Run Decoder
        cur_input_ids = np.array([[bos_token_id]], dtype=np.int64)
        
        for _ in range(20):
            decoder_inputs = {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
            }
            logits = dec_session.run(None, decoder_inputs)[0]
            
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
            
            if next_token_id == sep_token_id:
                break
            
            cur_input_ids = np.concatenate([cur_input_ids, [[next_token_id]]], axis=1)
        
        onnx_captions.append(processor.decode(cur_input_ids[0], skip_special_tokens=True))

    # Validate
    print("\n--- Results Comparison ---")
    correct = 0
    for i, (pt, onnx) in enumerate(zip(pt_captions, onnx_captions)):
        print(f"Image {i+1}:")
        print(f"  PyTorch: {pt}")
        print(f"  ONNX:    {onnx}")
        if pt == onnx:
            print("  Status:  MATCH ✅")
            correct += 1
        else:
            print("  Status:  MISMATCH ❌")
            
    if correct == len(pt_captions):
        print("\nAll Validations Passed.")

if __name__ == "__main__":
    run_validation()
