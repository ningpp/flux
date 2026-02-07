import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor
import os

# Config
MODEL_DIR = r"D:\models\onnx\blip-image-captioning-base"
PROCESSOR_PATH = r"D:\models\blip-image-captioning-base"
IMAGE_PATH = r"D:\tmp\img-2026-02-07-120114.png"

def main():
    if not os.path.exists(MODEL_DIR):
        print(f"Model dir not found: {MODEL_DIR}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return

    print("Loading processor...")
    try:
        processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
    except:
        # Fallback if local path invalid, try to load from hub or assume tokenizer.json is in model dir
        # For this script, let's assume the user has the processor config locally as implied
        print(f"Could not load processor from {PROCESSOR_PATH}, trying {MODEL_DIR}")
        processor = BlipProcessor.from_pretrained(MODEL_DIR)

    print("Loading ONNX models...")
    encoder_path = os.path.join(MODEL_DIR, "blip_vision_encoder.onnx")
    decoder_path = os.path.join(MODEL_DIR, "blip_text_decoder.onnx")
    
    sess_options = ort.SessionOptions()
    # sess_options.log_severity_level = 3
    
    encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=['CPUExecutionProvider'])

    print(f"Processing image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert('RGB')
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.numpy() # [1, 3, 384, 384]

    # Encoder
    encoder_outputs = encoder_session.run(None, {'pixel_values': pixel_values})
    encoder_hidden_states = encoder_outputs[0]

    # Decoder
    bos_token_id = 30522
    sep_token_id = 102
    
    cur_input_ids = np.array([[bos_token_id]], dtype=np.int64)
    
    print("Generating...")
    for i in range(20):
        out = decoder_session.run(None, {
            'input_ids': cur_input_ids,
            'encoder_hidden_states': encoder_hidden_states
        })
        logits = out[0] # [batch, seq_len, vocab]
        next_token_logits = logits[:, -1, :]
        next_token = np.argmax(next_token_logits, axis=-1)
        
        if next_token[0] == sep_token_id:
            break
            
        cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)

    caption = processor.batch_decode(cur_input_ids, skip_special_tokens=True)[0]
    print(f"Python Result: {caption}")

if __name__ == "__main__":
    main()
