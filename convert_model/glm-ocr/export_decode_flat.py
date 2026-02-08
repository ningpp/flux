"""
Export GLM-OCR LLM decode model using the "Tensor-Flat" wrapper pattern.

This approach flattens DynamicCache into individual tensors for ONNX inputs/outputs
and reconstructs the cache object inside the model wrapper.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, DynamicCache
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = r"D:\models\GLM-OCR"
OUTPUT_DIR = r"D:\models\onnx\GLM-OCR-LLM"


class GLMOcrLLMDecodeFlatKV(nn.Module):
    """
    Wrapper for GLM-OCR LLM decode with flattened KV cache tensors.
    
    Instead of passing DynamicCache object, we pass individual tensors:
    - past_key_0, past_value_0, past_key_1, past_value_1, ...
    
    Inside forward(), we reconstruct DynamicCache from these tensors.
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
        # Flattened past KV cache - 32 tensors (16 layers x 2 for key/value)
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
        Args:
            inputs_embeds: [batch_size, 1, hidden_size]
            attention_mask: [batch_size, total_seq_len]
            position_ids: [3, batch_size, 1] for M-RoPE
            past_key_i: [batch_size, num_kv_heads, past_seq_len, head_dim]
            past_value_i: [batch_size, num_kv_heads, past_seq_len, head_dim]
            
        Returns:
            logits: [batch_size, 1, vocab_size]
            present_key_0...15: Updated keys [batch_size, num_kv_heads, total_seq_len, head_dim]
            present_value_0...15: Updated values
        """
        # Collect all past tensors into a list
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
        
        # Reconstruct DynamicCache from flat tensors
        past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            past_key_values.update(
                past_keys[layer_idx],
                past_values[layer_idx],
                layer_idx
            )
        
        # Call language model with reconstructed cache
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        hidden_states = outputs.last_hidden_state
        new_past_key_values = outputs.past_key_values
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Flatten the new KV cache for output - using layers[i].keys/values
        result = [logits]
        for layer_idx in range(self.num_layers):
            result.append(new_past_key_values.layers[layer_idx].keys)
            result.append(new_past_key_values.layers[layer_idx].values)
            
        return tuple(result)


def export_decode_with_flat_kv(model, output_dir: str):
    """Export decode model using the Tensor-Flat pattern."""
    print("=" * 60)
    print("Exporting decode model with Tensor-Flat KV cache pattern...")
    print("=" * 60)
    
    wrapper = GLMOcrLLMDecodeFlatKV(model)
    wrapper.eval()
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    print(f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")
    
    batch_size = 1
    past_seq_len = 32
    
    # Prepare inputs
    inputs_embeds = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[[past_seq_len]], [[past_seq_len]], [[past_seq_len]]], dtype=torch.long)
    
    # Prepare flat past KV tensors
    past_kv_tensors = []
    for _ in range(num_layers):
        past_key = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_value = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_kv_tensors.extend([past_key, past_value])
    
    # Build input/output names and dynamic axes
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "inputs_embeds": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "total_seq_len"},
        "position_ids": {1: "batch_size"},
        "logits": {0: "batch_size"},
    }
    
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        
        dynamic_axes[f"past_key_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch_size", 2: "total_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch_size", 2: "total_seq_len"}
    
    print(f"Input names: {len(input_names)} inputs")
    print(f"Output names: {len(output_names)} outputs")
    
    # Test forward pass first
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            test_output = wrapper(
                inputs_embeds, attention_mask, position_ids,
                *past_kv_tensors
            )
            print(f"  ✓ Forward pass successful!")
            print(f"  Output logits shape: {test_output[0].shape}")
            print(f"  Number of outputs: {len(test_output)}")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            raise
    
    output_path = os.path.join(output_dir, "llm_decode.onnx")
    
    print(f"\nExporting to ONNX...")
    
    # Try dynamo_export first
    try:
        from torch.onnx import dynamo_export, ExportOptions
        
        export_options = ExportOptions(dynamic_shapes=True)
        
        all_inputs = (inputs_embeds, attention_mask, position_ids) + tuple(past_kv_tensors)
        
        onnx_program = dynamo_export(
            wrapper,
            *all_inputs,
            export_options=export_options,
        )
        onnx_program.save(output_path)
        print(f"  ✓ Saved with dynamo_export to {output_path}")
        
    except Exception as e:
        print(f"  dynamo_export failed: {e}")
        print("  Trying classic torch.onnx.export...")
        
        # Fall back to classic export
        all_inputs = (inputs_embeds, attention_mask, position_ids) + tuple(past_kv_tensors)
        
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
        print(f"  ✓ Saved with classic export to {output_path}")
    
    return output_path


def validate_decode_onnx(model, output_dir: str):
    """Validate the exported decode ONNX model."""
    import onnxruntime as ort
    import numpy as np
    
    print("\n" + "=" * 60)
    print("Validating decode ONNX model...")
    print("=" * 60)
    
    onnx_path = os.path.join(output_dir, "llm_decode.onnx")
    if not os.path.exists(onnx_path):
        # Check for external data file
        if os.path.exists(onnx_path + ".data"):
            pass
        else:
            print(f"  ✗ ONNX file not found: {onnx_path}")
            return
    
    # Load ONNX session
    print("Loading ONNX session...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Print input/output info
    print("\nONNX Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    
    print("\nONNX Outputs:")
    for out in session.get_outputs()[:5]:  # First 5
        print(f"  {out.name}: {out.shape} ({out.type})")
    print(f"  ... ({len(session.get_outputs())} total)")
    
    # Prepare test inputs
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    batch_size = 1
    past_seq_len = 64
    
    inputs_embeds = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[[past_seq_len]], [[past_seq_len]], [[past_seq_len]]], dtype=torch.long)
    
    past_kv_tensors = []
    for _ in range(num_layers):
        past_key = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_value = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_kv_tensors.extend([past_key, past_value])
    
    # PyTorch forward
    print("\nRunning PyTorch forward...")
    wrapper = GLMOcrLLMDecodeFlatKV(model)
    wrapper.eval()
    
    with torch.no_grad():
        pt_outputs = wrapper(inputs_embeds, attention_mask, position_ids, *past_kv_tensors)
    pt_logits = pt_outputs[0].numpy()
    
    # ONNX forward
    print("Running ONNX forward...")
    onnx_inputs = {
        'inputs_embeds': inputs_embeds.numpy(),
        'attention_mask': attention_mask.numpy(),
        'position_ids': position_ids.numpy(),
    }
    for i in range(num_layers):
        onnx_inputs[f'past_key_{i}'] = past_kv_tensors[i * 2].numpy()
        onnx_inputs[f'past_value_{i}'] = past_kv_tensors[i * 2 + 1].numpy()
    
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_logits = onnx_outputs[0]
    
    # Compare
    max_diff = np.max(np.abs(pt_logits - onnx_logits))
    mean_diff = np.mean(np.abs(pt_logits - onnx_logits))
    
    print(f"\nLogits comparison:")
    print(f"  PyTorch shape: {pt_logits.shape}")
    print(f"  ONNX shape: {onnx_logits.shape}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    
    if max_diff < 1e-3:
        print(f"  ✓ VALIDATION PASSED")
    else:
        print(f"  ✗ VALIDATION FAILED (diff too high)")
    
    # Also compare KV cache outputs
    print("\nKV cache output comparison (first layer):")
    pt_present_key_0 = pt_outputs[1].numpy()
    onnx_present_key_0 = onnx_outputs[1]
    kv_diff = np.max(np.abs(pt_present_key_0 - onnx_present_key_0))
    print(f"  present_key_0 max diff: {kv_diff:.2e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading GLM-OCR model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("Model loaded successfully!\n")
    
    # Export decode model
    export_decode_with_flat_kv(model, OUTPUT_DIR)
    
    # Validate
    validate_decode_onnx(model, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
