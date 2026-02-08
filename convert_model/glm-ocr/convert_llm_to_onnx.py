"""
Convert GLM-OCR LLM (Language Model) part to ONNX format using torch.onnx.dynamo_export.

This script exports:
- embedding.onnx: Token embedding layer
- llm_prefill.onnx: Full prefill model (processes entire sequence without KV cache)
- llm_decode.onnx: Decode model with KV cache for autoregressive generation
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoConfig
from pathlib import Path
import onnx
import numpy as np
import warnings
from typing import Tuple, Optional, List


MODEL_PATH = r"D:\models\GLM-OCR"
OUTPUT_DIR = r"D:\models\onnx\GLM-OCR-LLM"

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class GLMOcrEmbeddingWrapper(nn.Module):
    """Wrapper for the token embedding layer."""
    
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class GLMOcrLLMPrefill(nn.Module):
    """
    Wrapper for GLM-OCR LLM prefill (no KV cache).
    Takes inputs_embeds and generates logits + KV cache.
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
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            position_ids: [3, batch_size, seq_len] for M-RoPE
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            present_key_0...15: [batch_size, num_kv_heads, seq_len, head_dim]
            present_value_0...15: [batch_size, num_kv_heads, seq_len, head_dim]
        """
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        
        hidden_states = outputs[0]
        past_key_values = outputs[1]
        
        logits = self.lm_head(hidden_states)
        
        # Flatten KV cache for output
        result = [logits]
        for layer_kv in past_key_values:
            result.append(layer_kv[0])  # key
            result.append(layer_kv[1])  # value
            
        return tuple(result)


class GLMOcrLLMDecode(nn.Module):
    """
    Wrapper for GLM-OCR LLM decode with KV cache.
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
        *past_key_values_flat
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            inputs_embeds: [batch_size, 1, hidden_size]
            attention_mask: [batch_size, total_seq_len]
            position_ids: [3, batch_size, 1] for M-RoPE
            past_key_values_flat: Flattened past KV cache
            
        Returns:
            logits: [batch_size, 1, vocab_size]
            present_key_0...15: Updated keys
            present_value_0...15: Updated values
        """
        # Reconstruct past_key_values
        past_key_values = []
        for i in range(self.num_layers):
            past_key = past_key_values_flat[i * 2]
            past_value = past_key_values_flat[i * 2 + 1]
            past_key_values.append((past_key, past_value))
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        hidden_states = outputs[0]
        new_past_key_values = outputs[1]
        
        logits = self.lm_head(hidden_states)
        
        # Flatten KV cache for output
        result = [logits]
        for layer_kv in new_past_key_values:
            result.append(layer_kv[0])
            result.append(layer_kv[1])
            
        return tuple(result)


def export_embedding_layer(model, output_dir: str):
    """Export the token embedding layer to ONNX."""
    print("Exporting embedding layer...")
    
    wrapper = GLMOcrEmbeddingWrapper(model)
    wrapper.eval()
    
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    
    output_path = os.path.join(output_dir, "embedding.onnx")
    
    torch.onnx.export(
        wrapper,
        (input_ids,),
        output_path,
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "embeddings": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ✓ Saved to {output_path}")
    return output_path


def export_prefill_with_dynamo(model, output_dir: str):
    """Export the prefill model using torch.onnx.dynamo_export."""
    print("Exporting prefill model with dynamo_export...")
    
    wrapper = GLMOcrLLMPrefill(model)
    wrapper.eval()
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    
    batch_size = 1
    seq_len = 32
    
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1).clone()
    
    output_path = os.path.join(output_dir, "llm_prefill.onnx")
    
    # Use dynamo_export for better tracing support
    from torch.onnx import dynamo_export, ExportOptions
    
    export_options = ExportOptions(dynamic_shapes=True)
    
    onnx_program = dynamo_export(
        wrapper,
        inputs_embeds,
        attention_mask,
        position_ids,
        export_options=export_options,
    )
    onnx_program.save(output_path)
    
    print(f"  ✓ Saved to {output_path}")
    return output_path


def export_prefill_classic(model, output_dir: str):
    """Export the prefill model using classic torch.onnx.export."""
    print("Exporting prefill model (classic export)...")
    
    wrapper = GLMOcrLLMPrefill(model)
    wrapper.eval()
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    batch_size = 1
    seq_len = 32
    
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1).contiguous()
    
    # Build output names
    output_names = ["logits"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {1: "batch_size", 2: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }
    
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        dynamic_axes[f"present_key_{i}"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch_size", 2: "seq_len"}
    
    output_path = os.path.join(output_dir, "llm_prefill.onnx")
    
    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, position_ids),
        output_path,
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ✓ Saved to {output_path}")
    return output_path


def export_decode_classic(model, output_dir: str):
    """Export the decode model with KV cache."""
    print("Exporting decode model with KV cache...")
    
    wrapper = GLMOcrLLMDecode(model)
    wrapper.eval()
    
    config = model.config.text_config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    batch_size = 1
    past_seq_len = 32
    
    inputs_embeds = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[[past_seq_len]], [[past_seq_len]], [[past_seq_len]]], dtype=torch.long)
    
    # Past KV cache
    past_key_values_flat = []
    for _ in range(num_layers):
        past_key = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_value = torch.randn(batch_size, num_kv_heads, past_seq_len, head_dim, dtype=torch.float32)
        past_key_values_flat.extend([past_key, past_value])
    
    # Build names
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
    
    output_path = os.path.join(output_dir, "llm_decode.onnx")
    
    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, position_ids, *past_key_values_flat),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ✓ Saved to {output_path}")
    return output_path


def verify_onnx_model(model_path: str):
    """Verify the exported ONNX model."""
    print(f"  Verifying {os.path.basename(model_path)}...", end="")
    # For large models with external data, just check without loading data
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model, full_check=False)
    print(" ✓")
    return True


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
    )
    model.eval()
    
    print("\nModel config:")
    print(f"  Hidden size: {model.config.text_config.hidden_size}")
    print(f"  Num layers: {model.config.text_config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.text_config.num_attention_heads}")
    print(f"  Num KV heads: {model.config.text_config.num_key_value_heads}")
    print(f"  Head dim: {model.config.text_config.head_dim}")
    print(f"  Vocab size: {model.config.text_config.vocab_size}")
    
    print("\n" + "="*50)
    print("Exporting models to ONNX")
    print("="*50 + "\n")
    
    # Export models
    with torch.no_grad():
        embedding_path = export_embedding_layer(model, OUTPUT_DIR)
        
        # Try dynamo export first, fall back to classic
        try:
            prefill_path = export_prefill_with_dynamo(model, OUTPUT_DIR)
        except Exception as e:
            print(f"  Dynamo export failed: {e}")
            print("  Falling back to classic export...")
            prefill_path = export_prefill_classic(model, OUTPUT_DIR)
        
        try:
            decode_path = export_decode_classic(model, OUTPUT_DIR)
        except Exception as e:
            print(f"  Decode export failed: {e}")
            decode_path = None
    
    # Verify models
    print("\n" + "="*50)
    print("Verifying exported models")
    print("="*50 + "\n")
    
    verify_onnx_model(embedding_path)
    verify_onnx_model(prefill_path)
    if decode_path:
        verify_onnx_model(decode_path)
    
    print(f"\n✓ All models exported successfully to {OUTPUT_DIR}")
    print("\nExported files:")
    for f in os.listdir(OUTPUT_DIR):
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
