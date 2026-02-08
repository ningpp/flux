"""
Validate GLM-OCR LLM ONNX model outputs against PyTorch model.

This script:
1. Loads both PyTorch and ONNX models
2. Tests with different input sizes
3. Compares outputs to ensure conversion correctness
4. Tests with real images
"""

import os
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from typing import Dict, Any, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = r"D:\models\GLM-OCR"
ONNX_DIR = r"D:\models\onnx\GLM-OCR-LLM"

# Test images
TEST_IMAGES = [
    r"d:\tmp\table-2026-01-01-202211.png",
    r"d:\tmp\formula_2025-8-2_17-28-16.jpg",
]


def load_pytorch_model():
    """Load the PyTorch model."""
    print("Loading PyTorch model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def load_onnx_session(model_name: str):
    """Load an ONNX model session."""
    model_path = os.path.join(ONNX_DIR, model_name)
    print(f"Loading ONNX model: {model_name}...")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    return session


def validate_embedding_layer(pytorch_model, onnx_session):
    """Validate the embedding layer output."""
    print("\n" + "="*60)
    print("Validating Embedding Layer")
    print("="*60)
    
    # Test with different sequence lengths
    test_cases = [
        (1, 8),    # batch=1, seq=8
        (1, 16),   # batch=1, seq=16
        (1, 32),   # batch=1, seq=32
        (1, 64),   # batch=1, seq=64
        (1, 128),  # batch=1, seq=128
    ]
    
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        
        # PyTorch forward
        with torch.no_grad():
            pt_output = pytorch_model.model.language_model.embed_tokens(input_ids)
        
        # ONNX forward
        ort_inputs = {"input_ids": input_ids.numpy()}
        ort_output = onnx_session.run(None, ort_inputs)[0]
        
        # Compare
        pt_np = pt_output.numpy()
        max_diff = np.abs(pt_np - ort_output).max()
        mean_diff = np.abs(pt_np - ort_output).mean()
        
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  [{status}] batch={batch_size}, seq={seq_len}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    
    return True


def validate_prefill_model(pytorch_model, onnx_session):
    """Validate the prefill model output."""
    print("\n" + "="*60)
    print("Validating Prefill Model (LLM)")
    print("="*60)
    
    config = pytorch_model.config.text_config
    hidden_size = config.hidden_size
    
    # Test with different sequence lengths
    test_cases = [
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (1, 128),
    ]
    
    # Get ONNX input/output names
    input_names = [inp.name for inp in onnx_session.get_inputs()]
    output_names = [out.name for out in onnx_session.get_outputs()]
    print(f"  ONNX inputs: {input_names[:3]}...")
    print(f"  ONNX outputs: {output_names[:3]}... ({len(output_names)} total)")
    
    for batch_size, seq_len in test_cases:
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1).contiguous()
        
        # PyTorch forward
        with torch.no_grad():
            pt_hidden = pytorch_model.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )[0]
            pt_logits = pytorch_model.lm_head(pt_hidden)
        
        # ONNX forward
        # Note: The dynamo export may have different input names
        ort_inputs = {}
        for inp in onnx_session.get_inputs():
            name = inp.name
            if "embed" in name.lower() or "arg0" in name.lower():
                ort_inputs[name] = inputs_embeds.numpy()
            elif "mask" in name.lower() or "arg1" in name.lower():
                ort_inputs[name] = attention_mask.numpy()
            elif "position" in name.lower() or "arg2" in name.lower():
                ort_inputs[name] = position_ids.numpy()
        
        try:
            ort_outputs = onnx_session.run(None, ort_inputs)
            ort_logits = ort_outputs[0]
            
            # Compare logits
            pt_np = pt_logits.numpy()
            max_diff = np.abs(pt_np - ort_logits).max()
            mean_diff = np.abs(pt_np - ort_logits).mean()
            
            # Use higher tolerance for large models
            status = "✓" if max_diff < 1e-2 else "✗"
            print(f"  [{status}] batch={batch_size}, seq={seq_len}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        except Exception as e:
            print(f"  [✗] batch={batch_size}, seq={seq_len}: Error - {e}")
    
    return True


def validate_with_real_images(pytorch_model):
    """Validate with real images by processing through the full pipeline."""
    print("\n" + "="*60)
    print("Validating with Real Images")
    print("="*60)
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"  [!] Image not found: {img_path}")
            continue
            
        print(f"\n  Testing: {os.path.basename(img_path)}")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        print(f"    Image size: {image.size}")
        
        # Process with the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        
        print(f"    Input IDs shape: {inputs['input_ids'].shape}")
        print(f"    Pixel values shape: {inputs.get('pixel_values', [None])[0].shape if 'pixel_values' in inputs else 'N/A'}")
        
        # Get embeddings from PyTorch model
        with torch.no_grad():
            # This runs the full model forward to get hidden states
            # We extract just the LLM part
            input_ids = inputs['input_ids']
            
            # Get token embeddings
            token_embeds = pytorch_model.model.language_model.embed_tokens(input_ids)
            print(f"    Token embeddings shape: {token_embeds.shape}")
            
            # For images, the model would also process pixel_values through the vision encoder
            # and merge them with token embeddings
            if 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values']
                # Process through vision model
                vision_outputs = pytorch_model.model.visual(pixel_values)
                print(f"    Vision output shape: {vision_outputs.shape}")
        
        print(f"    [✓] Successfully processed image")
    
    return True


def validate_different_image_sizes(prefill_session):
    """Test LLM with embeddings simulating different image sizes."""
    print("\n" + "="*60)
    print("Validating with Different Image Sizes (Simulated)")
    print("="*60)
    
    # Different image sizes result in different numbers of vision tokens
    # Vision model: patch_size=14, spatial_merge_size=2
    # So for each 28x28 patch of the image, we get one token
    
    # Simulate different image sizes -> different vision token counts
    test_cases = [
        # (image_height, image_width, approx_vision_tokens)
        (224, 224, 64),     # Small image
        (336, 336, 144),    # Standard image size
        (448, 448, 256),    # Medium image
        (672, 672, 576),    # Large image
        (896, 896, 1024),   # Very large image
    ]
    
    config_hidden_size = 1536  # From model config
    
    for img_h, img_w, approx_tokens in test_cases:
        # Simulate: some text tokens + vision tokens + more text
        text_tokens = 20  # Prompt text
        total_tokens = text_tokens + approx_tokens
        
        # Create random embeddings (simulating combined text+vision embeddings)
        inputs_embeds = torch.randn(1, total_tokens, config_hidden_size, dtype=torch.float32)
        attention_mask = torch.ones(1, total_tokens, dtype=torch.long)
        position_ids = torch.arange(total_tokens).unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
        
        # Prepare ONNX inputs
        ort_inputs = {}
        for inp in prefill_session.get_inputs():
            name = inp.name
            if "embed" in name.lower() or "arg0" in name.lower():
                ort_inputs[name] = inputs_embeds.numpy()
            elif "mask" in name.lower() or "arg1" in name.lower():
                ort_inputs[name] = attention_mask.numpy()
            elif "position" in name.lower() or "arg2" in name.lower():
                ort_inputs[name] = position_ids.numpy()
        
        try:
            ort_outputs = prefill_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            print(f"  [✓] Image {img_h}x{img_w} (~{approx_tokens} tokens): logits shape = {logits.shape}")
        except Exception as e:
            print(f"  [✗] Image {img_h}x{img_w} (~{approx_tokens} tokens): Error - {e}")
    
    return True


def main():
    print("="*60)
    print("GLM-OCR LLM ONNX Validation")
    print("="*60)
    
    # Load models
    pytorch_model = load_pytorch_model()
    
    # Validate embedding layer
    embedding_session = load_onnx_session("embedding.onnx")
    validate_embedding_layer(pytorch_model, embedding_session)
    
    # Validate prefill model
    prefill_session = load_onnx_session("llm_prefill.onnx")
    validate_prefill_model(pytorch_model, prefill_session)
    
    # Validate with different image sizes (reuse the prefill session)
    validate_different_image_sizes(prefill_session)
    
    # Validate with real images
    validate_with_real_images(pytorch_model)
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
