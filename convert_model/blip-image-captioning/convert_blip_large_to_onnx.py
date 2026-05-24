import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnxruntime as ort
import numpy as np
import os
import shutil
import gc

# --- Configuration ---
MODEL_PATH = r"D:\models\blip-image-captioning-large"
ONNX_DIR = r"D:\models\onnx\blip-image-captioning-large"
IMAGE_PATHS = [
    r"D:\tmp\img-2026-02-07-120018.png",
    r"D:\tmp\img-2026-02-07-120114.png"
]

os.makedirs(ONNX_DIR, exist_ok=True)
ENCODER_ONNX_PATH = os.path.join(ONNX_DIR, "blip_vision_encoder.onnx")
DECODER_ONNX_PATH = os.path.join(ONNX_DIR, "blip_text_decoder.onnx")

# --- 1. Model Wrappers for Export ---

class BlipEncoderWrapper(nn.Module):
    """Wraps the vision model to return just the image embeddings."""
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        out = self.vision_model(pixel_values, return_dict=True)
        return out.last_hidden_state

class BlipDecoderWrapper(nn.Module):
    """Wraps the text decoder for autoregressive generation."""
    def __init__(self, text_decoder):
        super().__init__()
        self.text_decoder = text_decoder

    def forward(self, input_ids, encoder_hidden_states):
        out = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        return out.logits

# --- 2. Export Logic with Memory Management ---

def export_models():
    # --- Part A: Export Encoder ---
    print(f"Loading model (Pass 1 - Encoder) from {MODEL_PATH}...", flush=True)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()
    
    vis_config = model.config.vision_config
    image_size = vis_config.image_size
    patch_size = vis_config.patch_size
    encoder_hidden_size = vis_config.hidden_size
    
    # Extract Vision Model
    vision_model = model.vision_model
    # Remove reference to full model
    del model
    gc.collect()

    print(f"Exporting Vision Encoder to {ENCODER_ONNX_PATH}...", flush=True)
    encoder = BlipEncoderWrapper(vision_model)
    
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

    try:
        if os.path.exists(ENCODER_ONNX_PATH):
            os.remove(ENCODER_ONNX_PATH)
            
        torch.onnx.export(
            encoder,
            (dummy_pixel_values,),
            ENCODER_ONNX_PATH,
            input_names=['pixel_values'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={'pixel_values': {0: 'batch_size'}},
            opset_version=14
        )
        print("Vision Encoder Exported successfully.", flush=True)
    except Exception as e:
        print(f"Error exporting encoder: {e}", flush=True)
        return False

    # Cleanup Encoder
    del encoder
    del vision_model
    del dummy_pixel_values
    gc.collect()
    
    # --- Part B: Export Decoder ---
    print(f"Loading model (Pass 2 - Decoder) from {MODEL_PATH}...", flush=True)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()
    
    text_decoder = model.text_decoder
    # Remove reference to full model
    del model
    gc.collect()

    print(f"Exporting Text Decoder to {DECODER_ONNX_PATH}...", flush=True)
    decoder = BlipDecoderWrapper(text_decoder)
    
    # vis_seq_len = (384 // 16)^2 + 1 = 577
    vis_seq_len = (image_size // patch_size) ** 2 + 1
    
    dummy_input_ids = torch.randint(0, 100, (1, 10))
    dummy_encoder_hidden_states = torch.randn(1, vis_seq_len, encoder_hidden_size)

    try:
        if os.path.exists(DECODER_ONNX_PATH):
            os.remove(DECODER_ONNX_PATH)

        torch.onnx.export(
            decoder,
            (dummy_input_ids, dummy_encoder_hidden_states),
            DECODER_ONNX_PATH,
            input_names=['input_ids', 'encoder_hidden_states'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'encoder_hidden_states': {0: 'batch_size', 1: 'visual_sequence_length'}
            },
            opset_version=14
        )
        print("Text Decoder Exported successfully.", flush=True)
    except Exception as e:
        print(f"Error exporting decoder: {e}", flush=True)
        return False
        
    return True

# --- 3. Validation Logic ---

def run_validation():
    print("\n--- Running Validation ---", flush=True)
    
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    # We need the PyTorch model for ground truth, load it one last time
    # Or we can skip PyTorch generation if memory is tight, but validation is requested.
    # Let's try loading it.
    print("Loading PyTorch model for validation...", flush=True)
    try:
        pytorch_model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
        pytorch_model.eval()
    except Exception as e:
        print(f"Could not load PyTorch model for validation (OOM?): {e}")
        return

    # Load ONNX sessions
    try:
        enc_session = ort.InferenceSession(ENCODER_ONNX_PATH, providers=['CPUExecutionProvider'])
        dec_session = ort.InferenceSession(DECODER_ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX models: {e}")
        return

    # Process Images
    raw_images = []
    for p in IMAGE_PATHS:
        if os.path.exists(p):
            raw_images.append(Image.open(p).convert('RGB'))
            
    if not raw_images:
         print("No images found.")
         return

    inputs = processor(images=raw_images, return_tensors="pt")
    
    # 1. PyTorch Baseline
    print("Generating captions with PyTorch...", flush=True)
    with torch.no_grad():
        pt_out = pytorch_model.generate(**inputs, max_new_tokens=20)
        pt_captions = processor.batch_decode(pt_out, skip_special_tokens=True)

    # Free PyTorch model to ensure ONNX has RAM (though ONNX Runtime uses its own)
    del pytorch_model
    gc.collect()

    # 2. ONNX Inference
    print("Generating captions with ONNX...", flush=True)
    onnx_captions = []
    pixel_values_np = inputs.pixel_values.numpy()

    # Special tokens (hardcoded if model gone, but typically standard)
    # BLIP default ids
    bos_token_id = 30522 
    sep_token_id = 102

    for i in range(len(raw_images)):
        img_input = pixel_values_np[i:i+1]
        encoder_hidden_states = enc_session.run(None, {'pixel_values': img_input})[0]
        
        cur_input_ids = np.array([[bos_token_id]], dtype=np.int64)
        
        for _ in range(20):
            logits = dec_session.run(None, {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
            })[0]
            
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
            if next_token_id == sep_token_id:
                break
            cur_input_ids = np.concatenate([cur_input_ids, [[next_token_id]]], axis=1)
        
        onnx_captions.append(processor.decode(cur_input_ids[0], skip_special_tokens=True))

    # Validate
    for i, (pt, onnx) in enumerate(zip(pt_captions, onnx_captions)):
        print(f"Image {i+1}:")
        print(f"  PyTorch: {pt}")
        print(f"  ONNX:    {onnx}")
        if pt == onnx:
            print("  Status:  MATCH ✅")
        else:
            print("  Status:  MISMATCH ❌")

if __name__ == "__main__":
    if export_models():
        run_validation()
