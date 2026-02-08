"""
Complete Pipeline Validation for GLM-OCR ONNX Models

This script validates the complete inference pipeline:
  Vision Encoder → Embedding → LLM Prefill → LLM Decode

Validates with multiple image sizes and compares ONNX outputs against PyTorch.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = r"D:\models\GLM-OCR"
ONNX_DIR = r"D:\models\onnx\GLM-OCR-LLM"

REAL_IMAGES = [
    r"d:\tmp\table-2026-01-01-202211.png",
    r"d:\tmp\formula_2025-8-2_17-28-16.jpg",
]

SYNTHETIC_SIZES = [
    (224, 224),
    (336, 336),
    (448, 448),
    (336, 448),  # Non-square
]


# ============================================================
# Vision Encoder Wrapper (from export_vision_v2.py)
# ============================================================
class GLMOcrVisionEncoderONNX(nn.Module):
    """ONNX-exportable Vision Encoder wrapper."""
    
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
        
    def forward(self, pixel_values, pos_ids, max_grid_size):
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        for blk in self.blocks:
            hidden_states = self._block_forward(blk, hidden_states, position_embeddings)
        
        hidden_states = self.post_layernorm(hidden_states)
        
        # Downsample and merge - needs proper reshaping for conv2d
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.out_hidden_size)
        
        hidden_states = self.merger(hidden_states)
        return hidden_states
    
    def _block_forward(self, block, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = block.norm1(hidden_states)
        hidden_states = self._attention_forward(block.attn, hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = block.norm2(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    def _attention_forward(self, attn, hidden_states, position_embeddings):
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        qkv = attn.qkv(hidden_states)
        qkv = qkv.reshape(1, seq_length, 3, attn.num_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        cos, sin = position_embeddings
        cos = cos.view(1, 1, seq_length, -1)
        sin = sin.view(1, 1, seq_length, -1)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (attn.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(1, seq_length, -1)
        attn_output = attn.proj(attn_output)
        return attn_output.squeeze(0)
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ============================================================
# Helper: Compute pos_ids for vision encoder
# ============================================================
def compute_pos_ids(grid_thw, spatial_merge_size=2):
    """
    Pre-compute position IDs for rotary embeddings.
    This runs on CPU/Python and provides the pos_ids tensor for ONNX inference.
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
    max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_thw)
    return pos_ids, max_grid_size


# ============================================================
# Load Models
# ============================================================
def load_pytorch_model():
    """Load the original PyTorch model."""
    print("Loading PyTorch model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("  ✓ PyTorch model loaded")
    return model, processor


def load_onnx_sessions():
    """Load all ONNX sessions."""
    print("Loading ONNX sessions...")
    
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sessions = {}
    
    # Vision encoder
    vision_path = os.path.join(ONNX_DIR, "vision_encoder.onnx")
    sessions["vision"] = ort.InferenceSession(vision_path, opts, providers=["CPUExecutionProvider"])
    print(f"  ✓ vision_encoder.onnx loaded")
    
    # Embedding
    embed_path = os.path.join(ONNX_DIR, "embedding.onnx")
    sessions["embedding"] = ort.InferenceSession(embed_path, opts, providers=["CPUExecutionProvider"])
    print(f"  ✓ embedding.onnx loaded")
    
    # LLM Prefill
    prefill_path = os.path.join(ONNX_DIR, "llm_prefill.onnx")
    sessions["prefill"] = ort.InferenceSession(prefill_path, opts, providers=["CPUExecutionProvider"])
    print(f"  ✓ llm_prefill.onnx loaded")
    
    # LLM Decode
    decode_path = os.path.join(ONNX_DIR, "llm_decode.onnx")
    sessions["decode"] = ort.InferenceSession(decode_path, opts, providers=["CPUExecutionProvider"])
    print(f"  ✓ llm_decode.onnx loaded")
    
    # LLM Unified (optional - for memory-efficient decode)
    unified_path = os.path.join(ONNX_DIR, "llm_unified.onnx")
    if os.path.exists(unified_path):
        sessions["unified"] = ort.InferenceSession(unified_path, opts, providers=["CPUExecutionProvider"])
        print(f"  ✓ llm_unified.onnx loaded")
    
    return sessions


# ============================================================
# PyTorch Inference
# ============================================================
def pytorch_inference(model, processor, image, vision_wrapper, prompt="OCR:"):
    """Run complete inference with PyTorch model."""
    
    # Prepare inputs
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Get vision embeddings using the wrapper
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        pos_ids, max_grid_size = compute_pos_ids(image_grid_thw)
        max_grid_size_tensor = torch.tensor(max_grid_size, dtype=torch.long)
        
        vision_output = vision_wrapper(pixel_values, pos_ids, max_grid_size_tensor)
        
        # Get input embeddings
        input_ids = inputs["input_ids"]
        token_embeddings = model.model.language_model.embed_tokens(input_ids)
        
        # Merge image and text embeddings
        image_token_id = model.config.image_token_id
        image_mask = input_ids == image_token_id
        
        # Create merged embeddings
        inputs_embeds = token_embeddings.clone()
        inputs_embeds[image_mask] = vision_output.view(-1, vision_output.shape[-1])
        
        # Get position_ids from inputs or create them
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            # Create position IDs with M-RoPE
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, batch, seq]
        
        prefill_output = model.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        
        logits = model.lm_head(prefill_output.last_hidden_state)
        past_key_values = prefill_output.past_key_values
        
        # Decode one step
        next_token = logits[:, -1:, :].argmax(dim=-1)
        next_embeds = model.model.language_model.embed_tokens(next_token)
        
        # Update position IDs for decode
        next_pos = position_ids[:, :, -1:] + 1
        
        decode_output = model.model.language_model(
            inputs_embeds=next_embeds,
            position_ids=next_pos,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        
        decode_logits = model.lm_head(decode_output.last_hidden_state)
    
    return {
        "vision_output": vision_output.cpu().numpy(),
        "prefill_logits": logits.cpu().numpy(),
        "decode_logits": decode_logits.cpu().numpy(),
        "input_ids": input_ids.cpu().numpy(),
        "pixel_values": pixel_values.cpu().numpy(),
        "image_grid_thw": image_grid_thw.cpu().numpy() if isinstance(image_grid_thw, torch.Tensor) else np.array(image_grid_thw),
        "position_ids": position_ids.cpu().numpy(),
        "attention_mask": torch.ones_like(input_ids).cpu().numpy(),
    }


# ============================================================
# ONNX Inference
# ============================================================
def onnx_inference(sessions, pytorch_results, model):
    """Run complete inference with ONNX models."""
    
    # 1. Vision Encoder
    pixel_values = pytorch_results["pixel_values"]
    grid_thw = pytorch_results["image_grid_thw"]
    
    # Compute pos_ids
    pos_ids, max_grid_size = compute_pos_ids(grid_thw)
    
    vision_inputs = {
        "pixel_values": pixel_values.astype(np.float32),
        "pos_ids": pos_ids.numpy().astype(np.int64),
        "max_grid_size": np.array(max_grid_size, dtype=np.int64),
    }
    vision_output = sessions["vision"].run(None, vision_inputs)[0]
    
    # 2. Embedding
    input_ids = pytorch_results["input_ids"]
    embed_inputs = {"input_ids": input_ids.astype(np.int64)}
    token_embeddings = sessions["embedding"].run(None, embed_inputs)[0]
    
    # 3. Merge image and text embeddings
    image_token_id = model.config.image_token_id
    image_mask = input_ids == image_token_id
    
    inputs_embeds = token_embeddings.copy()
    inputs_embeds[image_mask] = vision_output.reshape(-1, vision_output.shape[-1])
    
    # 4. LLM Prefill
    position_ids = pytorch_results["position_ids"]
    attention_mask = pytorch_results["attention_mask"]
    
    prefill_inputs = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
        "position_ids": position_ids.astype(np.int64),
    }
    prefill_outputs = sessions["prefill"].run(None, prefill_inputs)
    prefill_logits = prefill_outputs[0]
    
    # Extract KV cache from prefill outputs
    # Output order: logits, then k0, v0, k1, v1, ... k15, v15
    kv_cache = {}
    for i in range(16):
        kv_cache[f"past_key_{i}"] = prefill_outputs[1 + i * 2]
        kv_cache[f"past_value_{i}"] = prefill_outputs[2 + i * 2]
    
    # 5. LLM Decode
    next_token = np.argmax(prefill_logits[:, -1:, :], axis=-1)
    
    # Get embedding for next token
    next_embed_inputs = {"input_ids": next_token.astype(np.int64)}
    next_embeds = sessions["embedding"].run(None, next_embed_inputs)[0]
    
    # Update position IDs
    next_pos = position_ids[:, :, -1:] + 1
    
    # Extend attention mask for decode (past_seq_len + 1)
    batch_size = attention_mask.shape[0]
    total_seq_len = attention_mask.shape[1] + 1
    decode_attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
    
    decode_inputs = {
        "inputs_embeds": next_embeds.astype(np.float32),
        "attention_mask": decode_attention_mask,
        "position_ids": next_pos.astype(np.int64),
        **{k: v.astype(np.float32) for k, v in kv_cache.items()},
    }
    decode_outputs = sessions["decode"].run(None, decode_inputs)
    decode_logits = decode_outputs[0]
    
    return {
        "vision_output": vision_output,
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
    }


# ============================================================
# ONNX Inference with Unified Model
# ============================================================
def onnx_inference_unified(sessions, pytorch_results, model):
    """Run complete inference with ONNX models using unified LLM for decode."""
    
    # 1. Vision Encoder
    pixel_values = pytorch_results["pixel_values"]
    grid_thw = pytorch_results["image_grid_thw"]
    
    # Compute pos_ids
    pos_ids, max_grid_size = compute_pos_ids(grid_thw)
    
    vision_inputs = {
        "pixel_values": pixel_values.astype(np.float32),
        "pos_ids": pos_ids.numpy().astype(np.int64),
        "max_grid_size": np.array(max_grid_size, dtype=np.int64),
    }
    vision_output = sessions["vision"].run(None, vision_inputs)[0]
    
    # 2. Embedding
    input_ids = pytorch_results["input_ids"]
    embed_inputs = {"input_ids": input_ids.astype(np.int64)}
    token_embeddings = sessions["embedding"].run(None, embed_inputs)[0]
    
    # 3. Merge image and text embeddings
    image_token_id = model.config.image_token_id
    image_mask = input_ids == image_token_id
    
    inputs_embeds = token_embeddings.copy()
    inputs_embeds[image_mask] = vision_output.reshape(-1, vision_output.shape[-1])
    
    # 4. LLM Prefill
    position_ids = pytorch_results["position_ids"]
    attention_mask = pytorch_results["attention_mask"]
    
    prefill_inputs = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
        "position_ids": position_ids.astype(np.int64),
    }
    prefill_outputs = sessions["prefill"].run(None, prefill_inputs)
    prefill_logits = prefill_outputs[0]
    
    # Extract KV cache from prefill outputs
    kv_cache = {}
    for i in range(16):
        kv_cache[f"past_key_{i}"] = prefill_outputs[1 + i * 2]
        kv_cache[f"past_value_{i}"] = prefill_outputs[2 + i * 2]
    
    # 5. LLM Decode using UNIFIED model
    next_token = np.argmax(prefill_logits[:, -1:, :], axis=-1)
    
    # Get embedding for next token
    next_embed_inputs = {"input_ids": next_token.astype(np.int64)}
    next_embeds = sessions["embedding"].run(None, next_embed_inputs)[0]
    
    # Update position IDs
    next_pos = position_ids[:, :, -1:] + 1
    
    # Extend attention mask for decode (past_seq_len + 1)
    batch_size = attention_mask.shape[0]
    total_seq_len = attention_mask.shape[1] + 1
    decode_attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
    
    decode_inputs = {
        "inputs_embeds": next_embeds.astype(np.float32),
        "attention_mask": decode_attention_mask,
        "position_ids": next_pos.astype(np.int64),
        **{k: v.astype(np.float32) for k, v in kv_cache.items()},
    }
    decode_outputs = sessions["unified"].run(None, decode_inputs)
    decode_logits = decode_outputs[0]
    
    return {
        "vision_output": vision_output,
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
    }


# ============================================================
# Validation
# ============================================================
def validate_outputs(pytorch_results, onnx_results, image_desc):
    """Compare PyTorch and ONNX outputs."""
    
    results = {}
    
    # Vision
    vision_diff = np.abs(pytorch_results["vision_output"] - onnx_results["vision_output"]).max()
    results["vision"] = vision_diff
    
    # Prefill logits
    prefill_diff = np.abs(pytorch_results["prefill_logits"] - onnx_results["prefill_logits"]).max()
    results["prefill"] = prefill_diff
    
    # Decode logits
    decode_diff = np.abs(pytorch_results["decode_logits"] - onnx_results["decode_logits"]).max()
    results["decode"] = decode_diff
    
    # Check pass/fail
    # Vision and decode should be very tight (1e-3)
    # Prefill can have higher tolerance (1e-1) due to error accumulation in logits
    vision_pass = vision_diff < 1e-3
    prefill_pass = prefill_diff < 1e-1  # Logits can have larger differences
    decode_pass = decode_diff < 1e-3
    all_pass = vision_pass and prefill_pass and decode_pass
    
    status = "[PASS]" if all_pass else "[FAIL]"
    print(f"  {image_desc}: {status}")
    print(f"    Vision: {vision_diff:.2e}, Prefill: {prefill_diff:.2e}, Decode: {decode_diff:.2e}")
    
    return all_pass, results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Complete Pipeline Validation")
    print("=" * 60)
    print()
    
    # Load models
    model, processor = load_pytorch_model()
    sessions = load_onnx_sessions()
    
    # Create vision wrapper
    vision_wrapper = GLMOcrVisionEncoderONNX(model.model.visual)
    vision_wrapper.eval()
    
    print()
    
    all_passed = True
    
    # Test with synthetic images
    print("-" * 60)
    print("Testing with Synthetic Images:")
    print("-" * 60)
    
    for width, height in SYNTHETIC_SIZES:
        # Create synthetic image
        image = Image.new("RGB", (width, height), color=(128, 128, 128))
        
        try:
            # PyTorch inference
            pytorch_results = pytorch_inference(model, processor, image, vision_wrapper)
            
            # ONNX inference
            onnx_results = onnx_inference(sessions, pytorch_results, model)
            
            # Validate
            passed, _ = validate_outputs(pytorch_results, onnx_results, f"{width}x{height}")
            all_passed = all_passed and passed
            
        except Exception as e:
            print(f"  {width}x{height}: [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print()
    
    # Test with real images
    print("-" * 60)
    print("Testing with Real Images:")
    print("-" * 60)
    
    for image_path in REAL_IMAGES:
        if not os.path.exists(image_path):
            print(f"  {os.path.basename(image_path)}: [SKIP] File not found")
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_desc = f"{os.path.basename(image_path)} ({image.size[0]}x{image.size[1]})"
            
            # PyTorch inference
            pytorch_results = pytorch_inference(model, processor, image, vision_wrapper)
            
            # ONNX inference
            onnx_results = onnx_inference(sessions, pytorch_results, model)
            
            # Validate
            passed, _ = validate_outputs(pytorch_results, onnx_results, image_desc)
            all_passed = all_passed and passed
            
        except Exception as e:
            print(f"  {os.path.basename(image_path)}: [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ ALL PIPELINE TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    # ========================================================
    # Test Unified Model Pipeline (if available)
    # ========================================================
    if "unified" in sessions:
        print()
        print("=" * 60)
        print("Unified Model Pipeline Validation")
        print("=" * 60)
        print()
        
        unified_passed = True
        
        # Test with synthetic images
        print("-" * 60)
        print("Testing Unified Pipeline with Synthetic Images:")
        print("-" * 60)
        
        for width, height in SYNTHETIC_SIZES:
            image = Image.new("RGB", (width, height), color=(128, 128, 128))
            
            try:
                pytorch_results = pytorch_inference(model, processor, image, vision_wrapper)
                onnx_results = onnx_inference_unified(sessions, pytorch_results, model)
                
                passed, _ = validate_outputs(pytorch_results, onnx_results, f"{width}x{height} (unified)")
                unified_passed = unified_passed and passed
                
            except Exception as e:
                print(f"  {width}x{height} (unified): [ERROR] {e}")
                import traceback
                traceback.print_exc()
                unified_passed = False
        
        print()
        
        # Test with real images
        print("-" * 60)
        print("Testing Unified Pipeline with Real Images:")
        print("-" * 60)
        
        for image_path in REAL_IMAGES:
            if not os.path.exists(image_path):
                print(f"  {os.path.basename(image_path)}: [SKIP] File not found")
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
                image_desc = f"{os.path.basename(image_path)} ({image.size[0]}x{image.size[1]}) (unified)"
                
                pytorch_results = pytorch_inference(model, processor, image, vision_wrapper)
                onnx_results = onnx_inference_unified(sessions, pytorch_results, model)
                
                passed, _ = validate_outputs(pytorch_results, onnx_results, image_desc)
                unified_passed = unified_passed and passed
                
            except Exception as e:
                print(f"  {os.path.basename(image_path)} (unified): [ERROR] {e}")
                import traceback
                traceback.print_exc()
                unified_passed = False
        
        print()
        print("=" * 60)
        if unified_passed:
            print("✓ ALL UNIFIED PIPELINE TESTS PASSED")
        else:
            print("✗ SOME UNIFIED TESTS FAILED")
        print("=" * 60)
        
        all_passed = all_passed and unified_passed
    
    return all_passed


if __name__ == "__main__":
    main()
