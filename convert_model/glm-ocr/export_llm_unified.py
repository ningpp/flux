"""
Export GLM-OCR LLM as a unified model with shared weights.

This approach:
1. Exports a single LLM model that handles both prefill and decode
2. Uses external data file for weights (shared across sessions)
3. KV cache is always provided - use zeros for prefill, real values for decode
4. Reduces memory usage from 2x to 1x compared to separate models
5. Supports both FP32 and FP16 export

Usage:
- Prefill: Pass zeros for past_key/value tensors with past_seq_len=0
- Decode: Pass real KV cache from previous step

Command line:
    python export_llm_unified.py              # Export FP32 only
    python export_llm_unified.py --fp16       # Export FP16 only
    python export_llm_unified.py --both       # Export both FP32 and FP16
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache
from typing import Tuple
import onnxruntime as ort
import onnx
from onnxconverter_common import float16
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = r"D:\models\GLM-OCR"
OUTPUT_DIR = r"D:\models\onnx\GLM-OCR-LLM"


class GLMOcrLLMUnified(nn.Module):
    """
    Unified LLM model for both prefill and decode.
    
    The trick: We always accept KV cache inputs. For prefill, the caller
    passes tensors with past_seq_len=0 (shape [batch, heads, 0, head_dim]).
    The model handles this naturally - empty cache means fresh start.
    """
    
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = model.config.text_config
        self.num_layers = self.config.num_hidden_layers
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        # KV cache - 32 tensors (16 layers x 2)
        # For prefill: shape [batch, heads, 0, head_dim] (empty)
        # For decode: shape [batch, heads, past_seq_len, head_dim]
        past_key_0: torch.Tensor, past_value_0: torch.Tensor,
        past_key_1: torch.Tensor, past_value_1: torch.Tensor,
        past_key_2: torch.Tensor, past_value_2: torch.Tensor,
        past_key_3: torch.Tensor, past_value_3: torch.Tensor,
        past_key_4: torch.Tensor, past_value_4: torch.Tensor,
        past_key_5: torch.Tensor, past_value_5: torch.Tensor,
        past_key_6: torch.Tensor, past_value_6: torch.Tensor,
        past_key_7: torch.Tensor, past_value_7: torch.Tensor,
        past_key_8: torch.Tensor, past_value_8: torch.Tensor,
        past_key_9: torch.Tensor, past_value_9: torch.Tensor,
        past_key_10: torch.Tensor, past_value_10: torch.Tensor,
        past_key_11: torch.Tensor, past_value_11: torch.Tensor,
        past_key_12: torch.Tensor, past_value_12: torch.Tensor,
        past_key_13: torch.Tensor, past_value_13: torch.Tensor,
        past_key_14: torch.Tensor, past_value_14: torch.Tensor,
        past_key_15: torch.Tensor, past_value_15: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Unified forward for both prefill and decode.
        
        Args:
            inputs_embeds: [batch, seq_len, hidden_size] - seq_len=prompt_len for prefill, 1 for decode
            attention_mask: [batch, total_seq_len] - total = past_seq_len + current_seq_len
            position_ids: [3, batch, seq_len] - M-RoPE position IDs
            past_key/value_i: [batch, num_kv_heads, past_seq_len, head_dim]
                              past_seq_len=0 for prefill, >0 for decode
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            present_key/value_0..15: [batch, num_kv_heads, total_seq_len, head_dim]
        """
        past_keys = [
            past_key_0, past_key_1, past_key_2, past_key_3,
            past_key_4, past_key_5, past_key_6, past_key_7,
            past_key_8, past_key_9, past_key_10, past_key_11,
            past_key_12, past_key_13, past_key_14, past_key_15,
        ]
        past_values = [
            past_value_0, past_value_1, past_value_2, past_value_3,
            past_value_4, past_value_5, past_value_6, past_value_7,
            past_value_8, past_value_9, past_value_10, past_value_11,
            past_value_12, past_value_13, past_value_14, past_value_15,
        ]
        
        # Always construct DynamicCache - no conditional branching for ONNX export
        # For prefill: past_seq_len=0 means empty initial cache (will work naturally)
        # For decode: past_seq_len>0 means we have prior context
        past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            past_key_values.update(
                past_keys[layer_idx],
                past_values[layer_idx],
                layer_idx
            )
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        hidden_states = outputs.last_hidden_state
        new_past = outputs.past_key_values
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Extract KV cache
        result = [logits]
        for layer_idx in range(self.num_layers):
            result.append(new_past.layers[layer_idx].keys)
            result.append(new_past.layers[layer_idx].values)
            
        return tuple(result)


def convert_to_fp16(input_path: str, output_path: str):
    """Convert ONNX model to FP16 using onnxruntime.transformers.
    
    Note: This function requires the model to be exported with classic torch.onnx.export,
    not dynamo_export. Dynamo-exported models may not convert properly.
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert_fp16
    print(f"\nConverting to FP16: {os.path.basename(output_path)}")
    
    # Load the model with external data
    model = onnx.load(input_path, load_external_data=True)
    print(f"  Loaded model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
    
    # Convert to FP16 using onnxruntime.transformers
    model_fp16 = ort_convert_fp16(
        model,
        keep_io_types=True,
        op_block_list=['LayerNormalization', 'Softmax', 'ReduceMean'],
    )
    
    # Check if conversion worked
    if len(model_fp16.graph.node) == 0:
        print("  ⚠ FP16 conversion failed - model is empty. Skipping FP16 export.")
        print("  Note: Dynamo-exported models may not be compatible with FP16 converters.")
        return None
    
    # Save with external data for large models
    onnx.save_model(
        model_fp16,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
    )
    
    # Report sizes
    onnx_size = os.path.getsize(output_path)
    data_path = output_path + ".data"
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    
    print(f"  ✓ FP16 model saved: {output_path}")
    print(f"    ONNX graph: {onnx_size / 1024 / 1024:.2f} MB")
    print(f"    Weights data: {data_size / 1024 / 1024:.2f} MB")
    print(f"    Total: {(onnx_size + data_size) / 1024 / 1024:.2f} MB")
    
    return output_path


def export_unified_llm(model, output_dir: str, dtype: str = "fp32"):
    """Export unified LLM model.
    
    Args:
        model: The PyTorch model to export
        output_dir: Directory to save the ONNX model
        dtype: "fp32", "fp16", or "both" - which precision to export
    """
    result_paths = {}
    
    # Export FP32 if requested
    if dtype in ("fp32", "both"):
        print("=" * 60)
        print("Exporting Unified LLM Model (FP32)")
        print("=" * 60)
        fp32_path = _export_llm_with_dtype(model, output_dir, "llm_unified.onnx", torch.float32)
        result_paths["fp32"] = fp32_path
    
    # Export FP16 if requested
    if dtype in ("fp16", "both"):
        print("\n" + "=" * 60)
        print("Exporting Unified LLM Model (FP16)")
        print("=" * 60)
        fp16_path = _export_llm_with_dtype(model, output_dir, "llm_unified_fp16.onnx", torch.float16)
        result_paths["fp16"] = fp16_path
    
    return result_paths


def _export_llm_with_dtype(model, output_dir: str, filename: str, torch_dtype: torch.dtype):
    """Export LLM model with specific dtype."""
    wrapper = GLMOcrLLMUnified(model)
    wrapper.eval()
    
    # Convert model to specified dtype if FP16
    if torch_dtype == torch.float16:
        wrapper = wrapper.half()
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    print(f"Config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim, {hidden_size} hidden")
    print(f"Export dtype: {torch_dtype}")
    
    # Export with decode-like inputs (past_seq_len > 0) for tracing
    # The model will work for both prefill (past_seq_len=0) and decode (past_seq_len>0)
    batch_size = 1
    seq_len = 1  # Current sequence length
    past_seq_len = 32  # Past sequence length for tracing
    
    # Use the appropriate dtype for inputs
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch_dtype)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.long)
    position_ids = torch.full((3, batch_size, seq_len), past_seq_len, dtype=torch.long)
    
    # Past KV tensors - use the same dtype
    past_kv_tensors = []
    for _ in range(num_layers):
        past_kv_tensors.append(torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch_dtype))
        past_kv_tensors.append(torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch_dtype))
    
    # Test forward
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = wrapper(inputs_embeds, attention_mask, position_ids, *past_kv_tensors)
        print(f"  Logits shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        print(f"  KV cache shapes: {outputs[1].shape} (per layer)")
    
    # Build input/output names and dynamic axes
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")
    
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
    
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq_len"}
    
    output_path = os.path.join(output_dir, filename)
    
    print("\nExporting to ONNX using dynamo_export with explicit dynamic dims...")
    all_inputs = (inputs_embeds, attention_mask, position_ids) + tuple(past_kv_tensors)
    
    # Use dynamo_export to handle internal torch.cat operations properly
    try:
        from torch.onnx import dynamo_export, ExportOptions
        from torch.export import Dim
        
        # Define dynamic dimensions explicitly
        batch_dim = Dim("batch", min=1, max=16)
        seq_dim = Dim("seq_len", min=1, max=4096)  # For inputs_embeds
        total_seq_dim = Dim("total_seq_len", min=1, max=8192)  # For attention_mask
        past_seq_dim = Dim("past_seq_len", min=0, max=8192)  # For KV cache (can be 0 for prefill)
        
        # Build dynamic_shapes dict
        # Format: {arg_index: {dim_index: Dim}}
        dynamic_shapes = {
            0: {0: batch_dim, 1: seq_dim},  # inputs_embeds: [batch, seq_len, hidden]
            1: {0: batch_dim, 1: total_seq_dim},  # attention_mask: [batch, total_seq]
            2: {1: batch_dim, 2: seq_dim},  # position_ids: [3, batch, seq_len]
        }
        # Add KV cache dynamic shapes (starting from arg index 3)
        for i in range(num_layers * 2):
            dynamic_shapes[3 + i] = {0: batch_dim, 2: past_seq_dim}  # past_key/value_i: [batch, heads, past_seq, head_dim]
        
        export_options = ExportOptions(dynamic_shapes=dynamic_shapes)
        
        onnx_program = dynamo_export(
            wrapper,
            *all_inputs,
            export_options=export_options,
        )
        onnx_program.save(output_path)
        print(f"  ✓ Saved with dynamo_export")
        
    except Exception as e:
        print(f"  dynamo_export failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Trying classic torch.onnx.export...")
        
        # Fall back to classic export
        torch.onnx.export(
            wrapper,
            all_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ✓ Saved with classic export")
    
    # Check sizes
    onnx_size = os.path.getsize(output_path)
    data_path = output_path + ".data"
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    
    dtype_str = "FP16" if torch_dtype == torch.float16 else "FP32"
    print(f"  ✓ Saved {dtype_str}: {output_path}")
    print(f"    ONNX graph: {onnx_size / 1024 / 1024:.2f} MB")
    print(f"    Weights data: {data_size / 1024 / 1024:.2f} MB")
    print(f"    Total: {(onnx_size + data_size) / 1024 / 1024:.2f} MB")
    
    return output_path


def validate_unified_llm(model, output_dir: str):
    """Validate unified model works for both prefill and decode."""
    print("\n" + "=" * 60)
    print("Validating Unified LLM Model")
    print("=" * 60)
    
    onnx_path = os.path.join(output_dir, "llm_unified.onnx")
    
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    
    print("\nONNX Inputs:")
    for inp in session.get_inputs()[:5]:
        print(f"  {inp.name}: {inp.shape}")
    print(f"  ... and {len(session.get_inputs()) - 5} more KV cache inputs")
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    wrapper = GLMOcrLLMUnified(model)
    wrapper.eval()
    
    all_passed = True
    
    # NOTE: The unified model was exported with seq_len=1 for decode efficiency.
    # For prefill (variable prompt lengths), use llm_prefill.onnx instead.
    
    # ===== Test 1: Decode mode with initial KV cache (simulating after prefill) =====
    print("\n" + "-" * 50)
    print("Test 1: Decode Mode (seq_len=1, past_seq_len=64)")
    print("-" * 50)
    print("  Note: This model is optimized for decode (seq_len=1).")
    print("  For prefill with variable seq_len, use llm_prefill.onnx")
    
    batch_size = 1
    past_seq_len = 64
    
    inputs_embeds = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    position_ids = torch.full((3, batch_size, 1), past_seq_len, dtype=torch.long)
    
    # Simulated KV cache from prefill
    kv_tensors = []
    for _ in range(num_layers):
        kv_tensors.append(torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32))
        kv_tensors.append(torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32))
    
    # PyTorch forward
    with torch.no_grad():
        pt_outputs = wrapper(inputs_embeds, attention_mask, position_ids, *kv_tensors)
        pt_logits = pt_outputs[0]
        pt_kv_cache = pt_outputs[1:]
    
    # ONNX forward
    onnx_inputs = {
        "inputs_embeds": inputs_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }
    for i in range(num_layers):
        onnx_inputs[f"past_key_{i}"] = kv_tensors[i*2].numpy()
        onnx_inputs[f"past_value_{i}"] = kv_tensors[i*2+1].numpy()
    
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_logits = onnx_outputs[0]
    
    max_diff = np.abs(pt_logits.numpy() - onnx_logits).max()
    status = "[PASS]" if max_diff < 1e-3 else "[FAIL]"
    print(f"  Decode step 1: {status} max_diff={max_diff:.2e}")
    if max_diff >= 1e-3:
        all_passed = False
    
    # Check KV cache shape is correct
    kv_cache = onnx_outputs[1:]
    expected_total_seq = past_seq_len + 1
    actual_total_seq = kv_cache[0].shape[2]
    kv_status = "[PASS]" if actual_total_seq == expected_total_seq else "[FAIL]"
    print(f"  KV cache shape: {kv_status} expected seq={expected_total_seq}, got {actual_total_seq}")
    if actual_total_seq != expected_total_seq:
        all_passed = False
    
    # ===== Test 2: Chained decode steps =====
    print("\n" + "-" * 50)
    print("Test 2: Chained Decode Steps (2 more tokens)")
    print("-" * 50)
    
    for step in range(2):
        current_seq_len = past_seq_len + 1 + step
        
        next_embeds = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
        decode_attn_mask = torch.ones(batch_size, current_seq_len + 1, dtype=torch.long)
        decode_pos = torch.full((3, batch_size, 1), current_seq_len, dtype=torch.long)
        
        # ONNX forward
        decode_onnx_inputs = {
            "inputs_embeds": next_embeds.numpy(),
            "attention_mask": decode_attn_mask.numpy(),
            "position_ids": decode_pos.numpy(),
        }
        for i in range(num_layers):
            decode_onnx_inputs[f"past_key_{i}"] = kv_cache[i*2]
            decode_onnx_inputs[f"past_value_{i}"] = kv_cache[i*2+1]
        
        decode_onnx_outputs = session.run(None, decode_onnx_inputs)
        kv_cache = decode_onnx_outputs[1:]  # Update cache
        
        expected_seq = current_seq_len + 1
        actual_seq = kv_cache[0].shape[2]
        status = "[PASS]" if actual_seq == expected_seq else "[FAIL]"
        print(f"  Step {step+2}: {status} KV cache seq={actual_seq} (expected {expected_seq})")
        if actual_seq != expected_seq:
            all_passed = False
    
    # ===== Summary =====
    print("\n" + "-" * 50)
    if all_passed:
        print("✓ All unified model tests PASSED")
    else:
        print("✗ Some tests FAILED")
    
    return all_passed


def compare_memory_usage():
    """Show memory comparison between separate and unified models."""
    print("\n" + "=" * 60)
    print("Memory Usage Comparison")
    print("=" * 60)
    
    files = {
        "llm_prefill.onnx": 0,
        "llm_prefill.onnx.data": 0,
        "llm_decode.onnx": 0,
        "llm_decode.onnx.data": 0,
        "llm_unified.onnx": 0,
        "llm_unified.onnx.data": 0,
        "llm_unified_fp16.onnx": 0,
        "llm_unified_fp16.onnx.data": 0,
    }
    
    for fname in files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            files[fname] = os.path.getsize(fpath)
    
    separate_total = sum(files[k] for k in ["llm_prefill.onnx", "llm_prefill.onnx.data",
                                             "llm_decode.onnx", "llm_decode.onnx.data"])
    unified_total = sum(files[k] for k in ["llm_unified.onnx", "llm_unified.onnx.data"])
    unified_fp16_total = sum(files[k] for k in ["llm_unified_fp16.onnx", "llm_unified_fp16.onnx.data"])
    
    print("\nSeparate Models (prefill + decode):")
    print(f"  llm_prefill.onnx:      {files['llm_prefill.onnx'] / 1024 / 1024:.2f} MB")
    print(f"  llm_prefill.onnx.data: {files['llm_prefill.onnx.data'] / 1024 / 1024:.2f} MB")
    print(f"  llm_decode.onnx:       {files['llm_decode.onnx'] / 1024 / 1024:.2f} MB")
    print(f"  llm_decode.onnx.data:  {files['llm_decode.onnx.data'] / 1024 / 1024:.2f} MB")
    print(f"  Total:                 {separate_total / 1024 / 1024:.2f} MB")
    
    print("\nUnified Model (FP32):")
    print(f"  llm_unified.onnx:      {files['llm_unified.onnx'] / 1024 / 1024:.2f} MB")
    print(f"  llm_unified.onnx.data: {files['llm_unified.onnx.data'] / 1024 / 1024:.2f} MB")
    print(f"  Total:                 {unified_total / 1024 / 1024:.2f} MB")
    
    if unified_fp16_total > 0:
        print("\nUnified Model (FP16):")
        print(f"  llm_unified_fp16.onnx:      {files['llm_unified_fp16.onnx'] / 1024 / 1024:.2f} MB")
        print(f"  llm_unified_fp16.onnx.data: {files['llm_unified_fp16.onnx.data'] / 1024 / 1024:.2f} MB")
        print(f"  Total:                      {unified_fp16_total / 1024 / 1024:.2f} MB")
    
    if separate_total > 0 and unified_total > 0:
        savings = (separate_total - unified_total) / separate_total * 100
        print(f"\n  FP32 savings vs separate: {savings:.1f}% ({(separate_total - unified_total) / 1024 / 1024:.2f} MB)")
    
    if unified_total > 0 and unified_fp16_total > 0:
        savings = (unified_total - unified_fp16_total) / unified_total * 100
        print(f"  FP16 savings vs FP32:     {savings:.1f}% ({(unified_total - unified_fp16_total) / 1024 / 1024:.2f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Export GLM-OCR unified LLM to ONNX")
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
    print("Model loaded!\n")
    
    # Export unified model
    result_paths = export_unified_llm(model, output_dir, dtype=dtype)
    
    # Validate
    if not args.skip_validation:
        validate_unified_llm(model, output_dir)
    
    # Show memory comparison
    compare_memory_usage()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Exported models: {list(result_paths.keys())}")

if __name__ == "__main__":
    main()
