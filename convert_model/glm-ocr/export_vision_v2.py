"""
Export GLM-OCR Vision Encoder to ONNX format with full dynamic shape support.

Strategy:
1. Pre-compute rotary position embeddings and cu_seqlens outside the model
2. Pass these as inputs to the vision model  
3. Use SDPA eager attention that doesn't require splitting by cu_seqlens

This allows the ONNX model to handle any image size with batch support.

Command line:
    python export_vision_v2.py              # Export FP32 only
    python export_vision_v2.py --fp16       # Export FP16 only
    python export_vision_v2.py --both       # Export both FP32 and FP16
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import onnxruntime as ort
import onnx
from onnxconverter_common import float16
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = r"D:\models\GLM-OCR"
OUTPUT_DIR = r"D:\models\onnx\GLM-OCR-LLM"


class GLMOcrVisionEncoderONNX(nn.Module):
    """
    ONNX-exportable Vision Encoder for GLM-OCR.
    
    This wrapper takes pre-computed position embeddings to avoid
    data-dependent control flow during tracing.
    
    For batch inference with different image sizes, images must be 
    processed sequentially (each as batch=1) since different sizes
    produce different sequence lengths.
    """
    
    def __init__(self, visual_model):
        super().__init__()
        self.patch_embed = visual_model.patch_embed
        self.rotary_pos_emb = visual_model.rotary_pos_emb
        self.blocks = visual_model.blocks
        self.post_layernorm = visual_model.post_layernorm
        self.downsample = visual_model.downsample
        self.merger = visual_model.merger
        self.spatial_merge_size = visual_model.spatial_merge_size
        self.out_hidden_size = visual_model.config.out_hidden_size
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_ids: torch.Tensor,
        max_grid_size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_patches, patch_features] - preprocessed patches
            pos_ids: [num_patches, 2] - position indices for rotary embeddings
            max_grid_size: scalar tensor - max(height_patches, width_patches)
            
        Returns:
            image_embeds: [num_output_tokens, 1536] - merged hidden states
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)
        
        # Compute rotary embeddings using pre-computed position indices
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # Process through transformer blocks
        # Note: We process the entire sequence without cu_seqlens splitting
        # This works for single-image inference
        for blk in self.blocks:
            hidden_states = self._block_forward(blk, hidden_states, position_embeddings)
        
        hidden_states = self.post_layernorm(hidden_states)
        
        # Downsample and merge
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.out_hidden_size)
        
        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states
    
    def _block_forward(self, blk, hidden_states, position_embeddings):
        """Forward through a single vision block without cu_seqlens."""
        # Self-attention
        residual = hidden_states
        hidden_states = blk.norm1(hidden_states)
        hidden_states = self._attention_forward(blk.attn, hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = blk.norm2(hidden_states)
        hidden_states = blk.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _attention_forward(self, attn, hidden_states, position_embeddings):
        """Attention forward without cu_seqlens splitting."""
        seq_length = hidden_states.shape[0]
        
        # QKV projection
        query_states, key_states, value_states = (
            attn.qkv(hidden_states)
            .reshape(seq_length, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        
        # RMS normalization
        query_states = attn.q_norm(query_states)
        key_states = attn.k_norm(key_states)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Reshape for attention: [batch=1, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        
        # Standard scaled dot-product attention (no GQA, no cu_seqlens)
        scale = attn.scaling
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = attn.proj(attn_output)
        
        return attn_output
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings to query and key tensors."""
        cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dim]
        
        # Split and rotate
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def compute_pos_ids(grid_thw: torch.Tensor, spatial_merge_size: int = 2):
    """
    Pre-compute position IDs for rotary embeddings.
    This runs on CPU/Python and provides the pos_ids tensor for ONNX inference.
    
    Args:
        grid_thw: [batch_size, 3] - (temporal, height_patches, width_patches) for each image
        spatial_merge_size: The spatial merge factor (default 2)
        
    Returns:
        pos_ids: [total_patches, 2] - position indices for all patches
        max_grid_size: int - maximum grid dimension
    """
    pos_ids_list = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
        
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        
        pos_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    
    pos_ids = torch.cat(pos_ids_list, dim=0)
    max_grid_size = int(grid_thw[:, 1:].max())
    
    return pos_ids, max_grid_size


def convert_to_fp16(input_path: str, output_path: str):
    """Convert ONNX model to FP16 using onnxconverter_common."""
    print(f"\nConverting to FP16: {os.path.basename(output_path)}")
    
    # Load the model
    model = onnx.load(input_path)
    
    # Convert to FP16, keeping certain ops in FP32 for stability
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs in original dtype for compatibility
        op_block_list=['LayerNormalization', 'Softmax', 'ReduceMean', 'Pow', 'Sqrt', 'Add'],
    )
    
    # Save the FP16 model
    onnx.save(model_fp16, output_path)
    
    # Report sizes
    file_size = os.path.getsize(output_path)
    print(f"  ✓ FP16 model saved: {output_path}")
    print(f"    File size: {file_size / 1024 / 1024:.2f} MB")
    
    return output_path


def export_vision_model(model, output_dir: str, dtype: str = "fp32"):
    """Export vision encoder to ONNX with dynamic shape support.
    
    Args:
        model: The PyTorch model to export
        output_dir: Directory to save the ONNX model
        dtype: "fp32", "fp16", or "both" - which precision to export
    """
    print("=" * 60)
    print(f"Exporting Vision Encoder (v2 - Dynamic Shapes, dtype={dtype})")
    print("=" * 60)
    
    wrapper = GLMOcrVisionEncoderONNX(model.model.visual)
    wrapper.eval()
    
    # Get processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Create sample image for tracing
    img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'test'}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors='pt')
    
    pixel_values = inputs.pixel_values
    grid_thw = inputs.image_grid_thw
    
    # Compute position IDs (this happens outside ONNX)
    pos_ids, max_grid_size = compute_pos_ids(grid_thw)
    max_grid_size_tensor = torch.tensor(max_grid_size, dtype=torch.long)
    
    print(f"Sample inputs:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw.tolist()}")
    print(f"  pos_ids: {pos_ids.shape}")
    print(f"  max_grid_size: {max_grid_size}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = wrapper(pixel_values, pos_ids, max_grid_size_tensor)
        print(f"  Output shape: {output.shape}")
    
    output_path = os.path.join(output_dir, "vision_encoder.onnx")
    result_paths = {}
    
    print("\nExporting to ONNX...")
    try:
        torch.onnx.export(
            wrapper,
            (pixel_values, pos_ids, max_grid_size_tensor),
            output_path,
            input_names=["pixel_values", "pos_ids", "max_grid_size"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "pos_ids": {0: "num_patches"},
                "image_embeds": {0: "num_output_tokens"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"  ✓ Saved FP32 to {output_path}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        result_paths["fp32"] = output_path
        
        # Convert to FP16 if requested
        if dtype in ("fp16", "both"):
            fp16_path = os.path.join(output_dir, "vision_encoder_fp16.onnx")
            convert_to_fp16(output_path, fp16_path)
            result_paths["fp16"] = fp16_path
        
        return result_paths
        
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_vision_model(model, output_dir: str):
    """Validate exported ONNX model with various image sizes."""
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
        (448, 336),  # Non-square
        (336, 448),  # Non-square other direction
    ]
    
    wrapper = GLMOcrVisionEncoderONNX(model.model.visual)
    wrapper.eval()
    
    print("\n" + "-" * 50)
    print("Testing with different image sizes:")
    print("-" * 50)
    
    all_passed = True
    for width, height in test_sizes:
        try:
            # Create test image
            img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'test'}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=[img], return_tensors='pt')
            
            pixel_values = inputs.pixel_values
            grid_thw = inputs.image_grid_thw
            
            # Compute position IDs
            pos_ids, max_grid_size = compute_pos_ids(grid_thw)
            max_grid_size_tensor = torch.tensor(max_grid_size, dtype=torch.long)
            
            # PyTorch forward
            with torch.no_grad():
                pt_output = wrapper(pixel_values, pos_ids, max_grid_size_tensor)
            
            # ONNX forward
            onnx_inputs = {
                'pixel_values': pixel_values.numpy(),
                'pos_ids': pos_ids.numpy(),
                'max_grid_size': np.array(max_grid_size, dtype=np.int64),
            }
            onnx_output = session.run(None, onnx_inputs)[0]
            
            # Compare
            max_diff = np.max(np.abs(pt_output.numpy() - onnx_output))
            
            status = "[PASS]" if max_diff < 1e-3 else "[FAIL]"
            if max_diff >= 1e-3:
                all_passed = False
            
            print(f"  {width}x{height}: {status} patches={pixel_values.shape[0]}, output_tokens={onnx_output.shape[0]}, max_diff={max_diff:.2e}")
            
        except Exception as e:
            print(f"  {width}x{height}: [ERROR] {e}")
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    
    return all_passed


def validate_with_real_images(model, output_dir: str):
    """Validate with real image files."""
    print("\n" + "=" * 60)
    print("Validating with Real Images...")
    print("=" * 60)
    
    onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    wrapper = GLMOcrVisionEncoderONNX(model.model.visual)
    wrapper.eval()
    
    real_images = [
        r"d:\tmp\table-2026-01-01-202211.png",
        r"d:\tmp\formula_2025-8-2_17-28-16.jpg",
    ]
    
    all_passed = True
    for img_path in real_images:
        if not os.path.exists(img_path):
            print(f"  {os.path.basename(img_path)}: [SKIP] File not found")
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'test'}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=[img], return_tensors='pt')
            
            pixel_values = inputs.pixel_values
            grid_thw = inputs.image_grid_thw
            pos_ids, max_grid_size = compute_pos_ids(grid_thw)
            max_grid_size_tensor = torch.tensor(max_grid_size, dtype=torch.long)
            
            # PyTorch
            with torch.no_grad():
                pt_output = wrapper(pixel_values, pos_ids, max_grid_size_tensor)
            
            # ONNX
            onnx_inputs = {
                'pixel_values': pixel_values.numpy(),
                'pos_ids': pos_ids.numpy(),
                'max_grid_size': np.array(max_grid_size, dtype=np.int64),
            }
            onnx_output = session.run(None, onnx_inputs)[0]
            
            max_diff = np.max(np.abs(pt_output.numpy() - onnx_output))
            status = "[PASS]" if max_diff < 1e-3 else "[FAIL]"
            if max_diff >= 1e-3:
                all_passed = False
            
            print(f"  {os.path.basename(img_path)}: {status} size={img.size}, max_diff={max_diff:.2e}")
            
        except Exception as e:
            print(f"  {os.path.basename(img_path)}: [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def parse_args():
    parser = argparse.ArgumentParser(description="Export GLM-OCR Vision Encoder to ONNX")
    parser.add_argument("--fp16", action="store_true", help="Export FP16 model only")
    parser.add_argument("--both", action="store_true", help="Export both FP32 and FP16 models")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to PyTorch model")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = args.output_dir
    model_path = args.model_path
    
    # Determine dtype
    if args.both:
        dtype = "both"
    elif args.fp16:
        dtype = "fp16"
    else:
        dtype = "fp32"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading GLM-OCR model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("Model loaded successfully!\n")
    
    # Export
    result_paths = export_vision_model(model, output_dir, dtype=dtype)
    
    if result_paths:
        if not args.skip_validation:
            # Validate with various sizes
            validate_vision_model(model, output_dir)
            
            # Validate with real images
            validate_with_real_images(model, output_dir)
    
    print("\n" + "=" * 60)
    print("DONE!")
    if result_paths:
        print(f"Exported models: {list(result_paths.keys())}")
