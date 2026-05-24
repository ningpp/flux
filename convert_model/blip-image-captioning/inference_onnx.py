import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor
import os
import time

# --- Configuration ---
# You can easily extend this map if needed
MODELS_CONFIG = {
    "base": {
        "onnx_dir": r"D:\models\onnx\blip-image-captioning-base",
        "processor_path": r"D:\models\blip-image-captioning-base"
    },
    "large": {
        "onnx_dir": r"D:\models\onnx\blip-image-captioning-large",
        "processor_path": r"D:\models\blip-image-captioning-large"
    }
}

IMAGES = [
    r"D:\tmp\img-2026-02-07-120018.png",
    r"D:\tmp\img-2026-02-07-120114.png"
]

class BlipOnnxInferencer:
    def __init__(self, model_key):
        config = MODELS_CONFIG.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")
            
        print(f"[{model_key}] Loading Processor from {config['processor_path']}...")
        self.processor = BlipProcessor.from_pretrained(config['processor_path'])
        
        encoder_path = os.path.join(config['onnx_dir'], "blip_vision_encoder.onnx")
        decoder_path = os.path.join(config['onnx_dir'], "blip_text_decoder.onnx")
        
        print(f"[{model_key}] Loading Vision Encoder ONNX: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
        
        print(f"[{model_key}] Loading Text Decoder ONNX: {decoder_path}")
        self.decoder_session = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
        
        # Determine special tokens
        # Typically bos=30522, sep=102 for BLIP text config (BertTokenizer)
        # Using processor.tokenizer to be sure
        self.bos_token_id = self.processor.tokenizer.bos_token_id if self.processor.tokenizer.bos_token_id is not None else 30522
        self.sep_token_id = self.processor.tokenizer.sep_token_id if self.processor.tokenizer.sep_token_id is not None else 102
        
    def generate_caption(self, image_path, max_length=20):
        if not os.path.exists(image_path):
            return f"Error: Image not found {image_path}"
            
        # Preprocess
        raw_image = Image.open(image_path).convert('RGB')
        # Use return_tensors="pt" and convert to numpy
        inputs = self.processor(images=raw_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.numpy() # Shape: (1, 3, H, W)
        
        # 1. Vision Encoder
        encoder_inputs = {'pixel_values': pixel_values}
        # Run encoder
        encoder_hidden_states = self.encoder_session.run(None, encoder_inputs)[0]
        
        # 2. Text Decoder Loop
        cur_input_ids = np.array([[self.bos_token_id]], dtype=np.int64)
        
        for step in range(max_length):
            decoder_inputs = {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
            }
            
            # Run decoder
            logits = self.decoder_session.run(None, decoder_inputs)[0]
            
            # Greedy search: argmax of the last token
            
            # Greedy search: argmax of the last token
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)[0]
            
            if next_token_id == self.sep_token_id:
                break
                
            # Append token
            cur_input_ids = np.concatenate([cur_input_ids, [[next_token_id]]], axis=1)
            
        # Decode text
        caption = self.processor.decode(cur_input_ids[0], skip_special_tokens=True)
        return caption

def main():
    print("=== BLIP ONNX Inference Demo ===")
    
    for model_name in ["base", "large"]:
        print(f"\n--- Initializing {model_name.upper()} Model ---")
        try:
            inferencer = BlipOnnxInferencer(model_name)
            
            print(f"\nRunning Inference on {len(IMAGES)} images...")
            for i, img_path in enumerate(IMAGES):
                start_time = time.time()
                caption = inferencer.generate_caption(img_path)
                end_time = time.time()
                
                print(f"Image {i+1}: {os.path.basename(img_path)}")
                print(f"Caption: {caption}")
                print(f"Time:    {end_time - start_time:.4f}s")
                print("-" * 30)
                
            # Cleanup to save memory for the next model
            del inferencer
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Failed to run for {model_name}: {e}")

if __name__ == "__main__":
    main()
