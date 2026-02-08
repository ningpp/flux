"""
Export GLM-OCR Vision Encoder to ONNX format.

The vision model takes:
- pixel_values: [num_patches, patch_features] - preprocessed image patches
- grid_thw: [batch_size, 3] - temporal, height, width grid dimensions

And outputs:
- image_embeds: [num_output_tokens, hidden_size] - visual embeddings for LLM

Note: The vision model has data-dependent control flow based on grid_thw,
which makes dynamic shape export difficult. We export with a fixed grid size
and the ONNX model will only work for that specific configuration.
For different image sizes, multiple ONNX models may be needed or
use PyTorch for preprocessing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = r"D:\models\GLM-OCR"
OUTPUT_DIR = r"D:\models\onnx\GLM-OCR-LLM"


class GLMOcrVisionWrapper(nn.Module):
    """
    Wrapper for GLM-OCR Vision Encoder.
    Returns only the last_hidden_state tensor.
    """
    
    def __init__(self, model):
        super().__init__()
        self.visual = model.model.visual
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_patches, patch_features]
            grid_thw: [batch_size, 3] - (temporal, height_patches, width_patches)
            
        Returns:
            image_embeds: [num_output_tokens, hidden_size]
        """
        output = self.visual(pixel_values, grid_thw=grid_thw)
        return output.last_hidden_state


def export_vision_model(model, output_dir: str):
    """Export vision encoder to ONNX."""
    print("=" * 60)
    print("Exporting Vision Encoder...")
    print("=" * 60)
    
    # Disable SDPA to avoid GQA export issues
    model.model.visual.set_attn_implementation("eager")
    
    wrapper = GLMOcrVisionWrapper(model)
    wrapper.eval()
    
    # Get processor to create proper inputs
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Create sample image for tracing - use a fixed size
    img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'test'}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors='pt')
    
    pixel_values = inputs.pixel_values
    grid_thw = inputs.image_grid_thw
    
    print(f"Sample input shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = wrapper(pixel_values, grid_thw)
        print(f"  Output shape: {output.shape}")
    
    output_path = os.path.join(output_dir, "vision_encoder.onnx")
    
    print("\nExporting to ONNX with classic export (fixed grid size)...")
    
    # Use classic export with fixed shapes (no dynamic axes for grid_thw)
    # This creates a model that works for 336x336 images (grid_thw=[1, 24, 24])
    try:
        torch.onnx.export(
            wrapper,
            (pixel_values, grid_thw),
            output_path,
            input_names=["pixel_values", "grid_thw"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                # grid_thw is fixed to avoid data-dependent control flow issues
                "image_embeds": {0: "num_output_tokens"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ✓ Saved to {output_path}")
        print(f"  Note: This model is exported for grid_thw={grid_thw.tolist()}")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return output_path


def validate_vision_model(model, output_dir: str):
    """Validate exported vision encoder."""
    import onnxruntime as ort
    
    print("\n" + "=" * 60)
    print("Validating Vision Encoder...")
    print("=" * 60)
    
    onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
    
    # Load ONNX session
    print("Loading ONNX session...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    print("\nONNX Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    
    print("\nONNX Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
    
    # Get processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Test with multiple image sizes
    test_sizes = [
        (224, 224),
        (336, 336),
        (448, 448),
        (560, 560),
        (672, 672),
    ]
    
    wrapper = GLMOcrVisionWrapper(model)
    wrapper.eval()
    
    print("\n" + "-" * 40)
    print("Testing with same image size as export (336x336):")
    print("-" * 40)
    print("Note: This model is exported for grid_thw=[[1, 24, 24]] (336x336 image)")
    print("      Different image sizes require separate model exports.")
    
    all_passed = True
    # Only test with the same size that was used for export
    test_sizes = [(336, 336)]
    
    for width, height in test_sizes:
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'test'}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors='pt')
        
        pixel_values = inputs.pixel_values
        grid_thw = inputs.image_grid_thw
        
        # PyTorch forward
        with torch.no_grad():
            pt_output = wrapper(pixel_values, grid_thw)
        
        # ONNX forward
        onnx_inputs = {
            'pixel_values': pixel_values.numpy(),
            'grid_thw': grid_thw.numpy(),
        }
        onnx_output = session.run(None, onnx_inputs)[0]
        
        # Compare
        max_diff = np.max(np.abs(pt_output.numpy() - onnx_output))
        
        status = "[PASS]" if max_diff < 1e-3 else "[FAIL]"
        if max_diff >= 1e-3:
            all_passed = False
        
        print(f"  {width}x{height}: {status} output_tokens={onnx_output.shape[0]}, max_diff={max_diff:.2e}")
    
    print("-" * 40)
    if all_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    
    return all_passed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading GLM-OCR model with eager attention (to avoid SDPA/GQA export issues)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # Avoid SDPA for ONNX export compatibility
    )
    model.eval()
    print("Model loaded successfully!\n")
    
    # Export vision model
    export_vision_model(model, OUTPUT_DIR)
    
    # Validate
    validate_vision_model(model, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
