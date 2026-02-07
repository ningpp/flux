"""
LLaVA-OneVision-Qwen2-0.5B Python ONNX inference script.
Used to validate Java implementation outputs.

Pipeline:
1. Preprocess image (resize to 384x384, normalize with mean=0.5 std=0.5)
2. Vision encoder (SigLIP) → image features [729, 1152]
3. Embed text tokens
4. Replace image token embeddings with vision features
5. Decoder with KV-cache → generate text
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import os
import sys

# Configuration
MODEL_DIR = r"D:\models\llava-onevision-qwen2-0.5b-ov-hf"
IMAGE_PATH = r"D:\tmp\img-2026-02-07-120114.png"
GPU_INDEX = -1  # -1 for CPU

# Special tokens
IM_START = 151644
IM_END = 151645
IMAGE_TOKEN = 151646

# Image processing constants
IMAGE_SIZE = 384
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


def preprocess_image(image_path):
    """
    Preprocess image to match SigLIP vision encoder requirements.
    Returns: pixel_values [1, 3, 384, 384] in CHW format
    """
    image = Image.open(image_path).convert('RGB')
    print(f"Original image size: {image.size}")
    
    # Resize to 384x384
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    
    # Convert to numpy array [H, W, C]
    img_array = np.array(image, dtype=np.float32)
    
    # Rescale to [0, 1]
    img_array = img_array * RESCALE_FACTOR
    
    # Normalize with mean=0.5, std=0.5
    img_array = (img_array - IMAGE_MEAN) / IMAGE_STD
    
    # Convert to CHW format [C, H, W]
    img_chw = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension [1, C, H, W]
    pixel_values = np.expand_dims(img_chw, axis=0)
    
    return pixel_values


def build_input_ids(tokenizer, num_image_tokens):
    """
    Build prompt input_ids with the LLaVA chat template.
    Format:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <image>
    Describe this image in detail.<|im_end|>
    <|im_start|>assistant
    """
    ids = []
    
    # <|im_start|>system
    ids.append(IM_START)
    system_text = tokenizer.encode("system\nYou are a helpful assistant.", add_special_tokens=False)
    ids.extend(system_text)
    ids.append(IM_END)
    
    # <|im_start|>user
    newline = tokenizer.encode("\n", add_special_tokens=False)
    ids.extend(newline)
    ids.append(IM_START)
    user_prefix = tokenizer.encode("user\n", add_special_tokens=False)
    ids.extend(user_prefix)
    
    # <image> token (single token, vision features will be 729 embeddings)
    ids.append(IMAGE_TOKEN)
    
    # Question
    user_text = tokenizer.encode("Describe this image in detail.", add_special_tokens=False)
    ids.extend(user_text)
    ids.append(IM_END)
    
    # <|im_start|>assistant
    ids.extend(newline)
    ids.append(IM_START)
    assistant_text = tokenizer.encode("assistant\n", add_special_tokens=False)
    ids.extend(assistant_text)
    
    return np.array([ids], dtype=np.int64)


def main():
    print("=" * 60)
    print("LLaVA-OneVision-Qwen2-0.5B — Python ONNX Inference")
    print("=" * 60)
    
    # Check paths
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found: {MODEL_DIR}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found: {IMAGE_PATH}")
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # Setup ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['CPUExecutionProvider']
    if GPU_INDEX >= 0:
        providers = [('CUDAExecutionProvider', {'device_id': GPU_INDEX})] + providers
    
    # Load ONNX models
    print("Loading ONNX models...")
    vision_encoder_path = os.path.join(MODEL_DIR, "onnx", "vision_encoder.onnx")
    embed_tokens_path = os.path.join(MODEL_DIR, "onnx", "embed_tokens.onnx")
    decoder_path = os.path.join(MODEL_DIR, "onnx", "decoder_model_merged.onnx")
    
    vision_session = ort.InferenceSession(vision_encoder_path, sess_options, providers=providers)
    embed_session = ort.InferenceSession(embed_tokens_path, sess_options, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
    
    print(f"Vision encoder inputs: {[inp.name for inp in vision_session.get_inputs()]}")
    print(f"Embed tokens inputs: {[inp.name for inp in embed_session.get_inputs()]}")
    print(f"Decoder inputs: {[inp.name for inp in decoder_session.get_inputs()]}")
    
    # 1. Preprocess image
    print(f"\nProcessing image: {IMAGE_PATH}")
    pixel_values = preprocess_image(IMAGE_PATH)
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Pixel values range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
    print(f"First 10 pixel values (channel 0, flattened): {pixel_values[0, 0, :, :].flatten()[:10]}")
    
    # 2. Vision encoder
    print("\nRunning vision encoder...")
    vision_outputs = vision_session.run(None, {'pixel_values': pixel_values})
    image_features = vision_outputs[0]  # [1, num_patches, hidden_size]
    print(f"Image features shape: {image_features.shape}")
    num_image_tokens = image_features.shape[1]
    vision_hidden_size = image_features.shape[2]
    print(f"Number of image tokens: {num_image_tokens}")
    print(f"Vision hidden size: {vision_hidden_size}")
    print(f"First 5 vision encoder output values: {image_features[0, 0, :5]}")
    
    # 3. Build input_ids
    print("\nBuilding input_ids...")
    input_ids = build_input_ids(tokenizer, num_image_tokens)
    seq_len = input_ids.shape[1]
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    print(f"Decoded prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    
    # 4. Embed tokens
    print("\nEmbedding tokens...")
    embed_outputs = embed_session.run(None, {'input_ids': input_ids})
    inputs_embeds = embed_outputs[0]  # [1, seq_len, hidden_size]
    print(f"Token embeddings shape: {inputs_embeds.shape}")
    text_hidden_size = inputs_embeds.shape[2]
    print(f"Text hidden size: {text_hidden_size}")
    
    # 5. Replace IMAGE_TOKEN embeddings with vision features
    print("\nMerging vision features...")
    inputs_embeds = inputs_embeds.copy()
    
    # Find IMAGE_TOKEN positions and replace with vision features
    image_token_positions = np.where(input_ids[0] == IMAGE_TOKEN)[0]
    print(f"IMAGE_TOKEN positions: {image_token_positions}")
    
    if len(image_token_positions) > 0:
        # The IMAGE_TOKEN is a single token but represents all vision features
        # We need to expand the sequence to accommodate all image tokens
        # For now, let's verify dimensions match
        if vision_hidden_size != text_hidden_size:
            print(f"WARNING: Vision hidden size ({vision_hidden_size}) != text hidden size ({text_hidden_size})")
            print("This may require a projection layer. Truncating/padding for now.")
            # Adjust vision features to match text hidden size
            if vision_hidden_size > text_hidden_size:
                image_features = image_features[:, :, :text_hidden_size]
            else:
                padding = np.zeros((1, num_image_tokens, text_hidden_size - vision_hidden_size), dtype=np.float32)
                image_features = np.concatenate([image_features, padding], axis=2)
        
        # Replace the IMAGE_TOKEN embedding with all vision features
        # This requires expanding the sequence
        pos = image_token_positions[0]
        before = inputs_embeds[:, :pos, :]
        after = inputs_embeds[:, pos+1:, :]
        inputs_embeds = np.concatenate([before, image_features, after], axis=1)
        
        # Update input_ids accordingly
        before_ids = input_ids[:, :pos]
        after_ids = input_ids[:, pos+1:]
        # Create placeholder IDs for image tokens (use IMAGE_TOKEN)
        image_token_ids = np.full((1, num_image_tokens), IMAGE_TOKEN, dtype=np.int64)
        input_ids = np.concatenate([before_ids, image_token_ids, after_ids], axis=1)
        
        seq_len = input_ids.shape[1]
        print(f"Updated sequence length: {seq_len}")
    
    print(f"Final inputs_embeds shape: {inputs_embeds.shape}")
    
    # Debug: print sample embedding values
    print(f"First 5 embedding values at position 0: {inputs_embeds[0, 0, :5]}")
    print(f"First 5 embedding values at position {pos} (first vision feature): {inputs_embeds[0, pos, :5]}")
    
    # 6. Create attention mask and position IDs
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    # 7. Run decoder with KV cache initialization
    print("\nRunning decoder...")
    try:
        # Check decoder input requirements
        decoder_input_names = [inp.name for inp in decoder_session.get_inputs()]
        print(f"Decoder expects inputs: {decoder_input_names}")
        
        # Get model configuration for KV cache
        import json
        with open(os.path.join(MODEL_DIR, "config.json"), 'r') as f:
            config = json.load(f)
        text_config = config.get('text_config', {})
        num_layers = text_config.get('num_hidden_layers', 24)
        num_kv_heads = text_config.get('num_key_value_heads', 2)
        num_attention_heads = text_config.get('num_attention_heads', 14)
        hidden_size = text_config.get('hidden_size', 896)
        head_dim = hidden_size // num_attention_heads
        
        print(f"Model config: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
        
        # Initialize decoder inputs with empty KV cache
        decoder_inputs = {
            'inputs_embeds': inputs_embeds.astype(np.float32),
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        # Initialize empty KV cache for each layer
        for layer_idx in range(num_layers):
            # Shape: [batch_size, num_kv_heads, 0, head_dim] - empty cache
            empty_cache = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
            decoder_inputs[f'past_key_values.{layer_idx}.key'] = empty_cache
            decoder_inputs[f'past_key_values.{layer_idx}.value'] = empty_cache
        
        print(f"Initialized KV cache with shape: [1, {num_kv_heads}, 0, {head_dim}]")
        
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]  # [batch_size, seq_len, vocab_size]
        
        print(f"Logits shape: {logits.shape}")
        
        # Generate tokens (greedy decoding from logits)
        # For simplicity, just take argmax of last position
        next_token_logits = logits[0, -1, :]
        next_token = np.argmax(next_token_logits)
        
        print(f"\nFirst generated token ID: {next_token}")
        print(f"First generated token: '{tokenizer.decode([next_token])}'")
        
        # Full generation would require a loop with KV cache updates
        print("\nNote: Full text generation requires implementing KV cache loop.")
        print("This script validates the first token generation matches Java implementation.")
        
    except Exception as e:
        print(f"Error running decoder: {e}")
        print("\nDecoder input details:")
        for inp in decoder_session.get_inputs():
            print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
        raise
    
    print("\n" + "=" * 60)
    print("Inference pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
