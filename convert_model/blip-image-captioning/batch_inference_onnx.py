import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor
import os
import time
import glob

# --- Configuration ---
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

# Source directory for batch processing
IMAGE_DIR = r"D:\tmp"
IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg"]

class BlipBatchOnnxInferencer:
    def __init__(self, model_key):
        config = MODELS_CONFIG.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")
            
        print(f"[{model_key}] Loading Processor...")
        self.processor = BlipProcessor.from_pretrained(config['processor_path'])
        
        encoder_path = os.path.join(config['onnx_dir'], "blip_vision_encoder.onnx")
        decoder_path = os.path.join(config['onnx_dir'], "blip_text_decoder.onnx")
        
        print(f"[{model_key}] Loading ONNX Sessions...")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
        self.decoder_session = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
        
        self.bos_token_id = self.processor.tokenizer.bos_token_id or 30522
        self.sep_token_id = self.processor.tokenizer.sep_token_id or 102
        
    def generate_batch(self, image_paths, max_new_tokens=20):
        """
        Performs batch inference. 
        Note: The current ONNX export supports dynamic batching.
        """
        raw_images = []
        valid_paths = []
        for p in image_paths:
            if os.path.exists(p):
                raw_images.append(Image.open(p).convert('RGB'))
                valid_paths.append(p)
        
        if not raw_images:
            return []

        # Preprocess batch
        inputs = self.processor(images=raw_images, return_tensors="pt")
        pixel_values = inputs.pixel_values.numpy()
        batch_size = len(raw_images)

        # 1. Vision Encoder (Run once for the whole batch)
        encoder_hidden_states = self.encoder_session.run(None, {'pixel_values': pixel_values})[0]
        
        # 2. Text Decoder Loop (Greedy)
        # Initialize input_ids for the whole batch: [batch_size, 1]
        cur_input_ids = np.full((batch_size, 1), self.bos_token_id, dtype=np.int64)
        finished = np.zeros(batch_size, dtype=bool)
        
        for _ in range(max_new_tokens):
            if finished.all():
                break
                
            decoder_inputs = {
                'input_ids': cur_input_ids,
                'encoder_hidden_states': encoder_hidden_states
            }
            
            logits = self.decoder_session.run(None, decoder_inputs)[0]
            
            # Predict next token for each sequence in batch
            next_tokens = np.argmax(logits[:, -1, :], axis=-1) # Shape: (batch_size,)
            
            # Update finished status
            finished |= (next_tokens == self.sep_token_id)
            
            # Force SEP for already finished sequences to prevent junk tail
            next_tokens[finished] = self.sep_token_id
            
            # Append next tokens
            cur_input_ids = np.concatenate([cur_input_ids, next_tokens[:, None]], axis=1)

        # Decode all
        captions = self.processor.batch_decode(cur_input_ids, skip_special_tokens=True)
        return list(zip(valid_paths, captions))

def main():
    # 1. Collect Images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    # Sort for consistent output
    image_files.sort()
    
    # Limit to a smaller subset for demonstration if there are too many
    # or process in chunks
    CHUNK_SIZE = 4
    image_files = image_files[:CHUNK_SIZE]
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} images for batch processing.")

    # 2. Process models
    for model_key in ["base", "large"]:
        print(f"\n{'='*50}")
        print(f"Starting Batch Inference with BLIP-{model_key.upper()}")
        print(f"{'='*50}")
        
        try:
            inferencer = BlipBatchOnnxInferencer(model_key)
            
            print(f"Processing batch of {len(image_files)} images...")
            start_time = time.time()
            # Process everything in one batch (if memory allows)
            results = inferencer.generate_batch(image_files)
            duration = time.time() - start_time
            
            print(f"\nResults for {model_key.upper()}:")
            for path, caption in results:
                print(f"[{os.path.basename(path)}]: {caption}")
            
            print(f"\nTotal time: {duration:.2f}s ({duration/len(image_files):.2f}s per image)")
            
            # Memory cleanup
            del inferencer
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error during {model_key} batch inference: {e}")

if __name__ == "__main__":
    main()
