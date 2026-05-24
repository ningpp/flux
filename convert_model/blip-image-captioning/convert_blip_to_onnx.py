import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnxruntime as ort
import numpy as np
import os
import shutil

# --- Configuration ---
MODEL_PATH = r"D:\models\blip-image-captioning-base"
ONNX_DIR = r"D:\models\onnx\blip-image-captioning-base"
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
        # attention_mask (encoder_attention_mask) is removed as it seems unused/pruned by export
        out = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        return out.logits

# --- 2. Export Logic ---

def export_models():
    print(f"Loading model from {MODEL_PATH}...")
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()

    # --- Cleanup ---
    if os.path.exists(ENCODER_ONNX_PATH):
        try:
            os.remove(ENCODER_ONNX_PATH)
        except: pass
    if os.path.exists(DECODER_ONNX_PATH):
        try:
            os.remove(DECODER_ONNX_PATH)
        except: pass

    # --- Export Encoder ---
    print(f"Exporting Vision Encoder to {ENCODER_ONNX_PATH}...")
    encoder = BlipEncoderWrapper(model.vision_model)
    
    image_size = model.config.vision_config.image_size
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        encoder,
        (dummy_pixel_values,),
        ENCODER_ONNX_PATH,
        input_names=['pixel_values'],
        output_names=['encoder_hidden_states'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}},
        opset_version=14
    )

    # --- Export Decoder ---
    print(f"Exporting Text Decoder to {DECODER_ONNX_PATH}...")
    decoder = BlipDecoderWrapper(model.text_decoder)
    
    vis_seq_len = (image_size // model.config.vision_config.patch_size) ** 2 + 1
    
    hidden_size = model.config.text_config.hidden_size
    
    dummy_input_ids = torch.randint(0, 100, (1, 10))
    dummy_encoder_hidden_states = torch.randn(1, vis_seq_len, hidden_size)
    # attention_mask removed

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
    
    print("Export Complete.")
    return processor, model

# --- 3. Validation Logic ---

def run_validation(processor, pytorch_model):
    print("\n--- Running Validation ---")
    
    # Load ONNX sessions
    try:
        enc_session = ort.InferenceSession(ENCODER_ONNX_PATH, providers=['CPUExecutionProvider'])
        dec_session = ort.InferenceSession(DECODER_ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX models: {e}")
        return

    # Process Images
    raw_images = []
    print(f"Loading images: {IMAGE_PATHS}")
    for p in IMAGE_PATHS:
        if os.path.exists(p):
            raw_images.append(Image.open(p).convert('RGB'))
        else:
            print(f"Warning: Image not found {p}")

    if not raw_images:
        print("No images found for validation.")
        return

    inputs = processor(images=raw_images, return_tensors="pt")
    
    # 1. PyTorch Baseline
    print("Generating captions with PyTorch (Original)...")
    with torch.no_grad():
        pt_out = pytorch_model.generate(**inputs, max_length=20)
        pt_captions = processor.batch_decode(pt_out, skip_special_tokens=True)

    # 2. ONNX Inference (Greedy)
    print("Generating captions with ONNX (Converted)...")
    onnx_captions = []
    pixel_values_np = inputs.pixel_values.numpy()

    # Get special tokens
    bos_token_id = pytorch_model.config.text_config.bos_token_id or 30522
    sep_token_id = pytorch_model.config.text_config.sep_token_id or 102

    for i in range(len(raw_images)):
        # -- Vision Encoder --
        img_input = pixel_values_np[i:i+1] # Batch size 1
        encoder_inputs = {'pixel_values': img_input}
        encoder_hidden_states = enc_session.run(None, encoder_inputs)[0]
        
        # -- Text Decoder Loop --
        cur_input_ids = np.array([[bos_token_id]], dtype=np.int64)
        
        # Prepare attention mask for encoder (all 1s for the image tokens)
        # Not used in ONNX inference as it was pruned

        for _ in range(20): # Max length
            decoder_inputs = {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
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
    proc, pt_model = export_models()
    run_validation(proc, pt_model)
