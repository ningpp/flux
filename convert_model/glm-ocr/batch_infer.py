"""
Batch Inference and Validation for GLM-OCR ONNX Models

This script tests ONNX models with different batch sizes and validates:
1. Output consistency across batch sizes (same image should produce identical tokens)
2. Performance metrics for different batch configurations
3. Full generation loop with KV cache

Usage:
    python batch_infer.py --image-dir d:/tmp/ocr_images --batch-sizes 1,2,4 --max-tokens 100
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = r"D:\models\GLM-OCR"
ONNX_DIR = r"D:\models\onnx\GLM-OCR-LLM"

NUM_LAYERS = 16
HIDDEN_SIZE = 1536
SPATIAL_MERGE_SIZE = 2


# ============================================================
# Data Classes
# ============================================================
@dataclass
class InferenceResult:
    """Result from a single inference."""
    image_path: str
    generated_tokens: List[int]
    generated_text: str
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_per_second: float


@dataclass
class BatchResult:
    """Result from a batch inference."""
    batch_size: int
    results: List[InferenceResult]
    total_time_ms: float
    avg_time_per_image_ms: float
    throughput_images_per_sec: float


# ============================================================
# Helper: Compute pos_ids for vision encoder
# ============================================================
def compute_pos_ids(grid_thw, spatial_merge_size=SPATIAL_MERGE_SIZE):
    """
    Pre-compute position IDs for rotary embeddings.
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
# ONNX Session Manager
# ============================================================
class ONNXSessionManager:
    """Manages ONNX sessions for all model components."""
    
    def __init__(self, onnx_dir: str, provider: str = "CPUExecutionProvider"):
        self.onnx_dir = onnx_dir
        self.provider = provider
        self.sessions = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """Load all ONNX sessions."""
        print("Loading ONNX sessions...")
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Vision encoder
        vision_path = os.path.join(self.onnx_dir, "vision_encoder.onnx")
        if os.path.exists(vision_path):
            self.sessions["vision"] = ort.InferenceSession(
                vision_path, opts, providers=[self.provider]
            )
            print(f"  ✓ vision_encoder.onnx loaded")
        else:
            print(f"  ✗ vision_encoder.onnx not found")
        
        # Embedding
        embed_path = os.path.join(self.onnx_dir, "embedding.onnx")
        if os.path.exists(embed_path):
            self.sessions["embedding"] = ort.InferenceSession(
                embed_path, opts, providers=[self.provider]
            )
            print(f"  ✓ embedding.onnx loaded")
        else:
            print(f"  ✗ embedding.onnx not found")
        
        # LLM Prefill
        prefill_path = os.path.join(self.onnx_dir, "llm_prefill.onnx")
        if os.path.exists(prefill_path):
            self.sessions["prefill"] = ort.InferenceSession(
                prefill_path, opts, providers=[self.provider]
            )
            print(f"  ✓ llm_prefill.onnx loaded")
        else:
            print(f"  ✗ llm_prefill.onnx not found")
        
        # LLM Decode
        decode_path = os.path.join(self.onnx_dir, "llm_decode.onnx")
        if os.path.exists(decode_path):
            self.sessions["decode"] = ort.InferenceSession(
                decode_path, opts, providers=[self.provider]
            )
            print(f"  ✓ llm_decode.onnx loaded")
        else:
            print(f"  ✗ llm_decode.onnx not found")
        
        print()
    
    def get_session(self, name: str) -> ort.InferenceSession:
        """Get a specific session by name."""
        if name not in self.sessions:
            raise ValueError(f"Session '{name}' not loaded. Available: {list(self.sessions.keys())}")
        return self.sessions[name]
    
    def has_session(self, name: str) -> bool:
        """Check if a session is available."""
        return name in self.sessions


# ============================================================
# Single Image Inference
# ============================================================
class GLMOcrONNXInference:
    """
    GLM-OCR ONNX inference engine.
    
    Handles:
    - Vision encoding (single image at a time due to variable sequence lengths)
    - Token embedding
    - LLM prefill with merged vision + text embeddings
    - Autoregressive decode with KV cache
    """
    
    def __init__(
        self, 
        session_manager: ONNXSessionManager,
        processor: AutoProcessor,
        image_token_id: int,
        eos_token_id: int,
        max_tokens: int = 256,
    ):
        self.session_mgr = session_manager
        self.processor = processor
        self.image_token_id = image_token_id
        self.eos_token_id = eos_token_id
        self.max_tokens = max_tokens
    
    def preprocess_image(self, image: Image.Image, prompt: str = "OCR:") -> Dict[str, np.ndarray]:
        """
        Preprocess image and text for inference.
        
        Returns dict with:
        - input_ids: [1, seq_len]
        - pixel_values: [num_patches, features]
        - image_grid_thw: [1, 3]
        - position_ids: [3, 1, seq_len]
        - attention_mask: [1, seq_len]
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        
        # Convert to numpy
        input_ids = inputs["input_ids"].numpy()
        pixel_values = inputs["pixel_values"].numpy()
        image_grid_thw = inputs["image_grid_thw"].numpy() if isinstance(
            inputs["image_grid_thw"], torch.Tensor
        ) else np.array(inputs["image_grid_thw"])
        
        # Get or create position_ids
        if "position_ids" in inputs:
            position_ids = inputs["position_ids"].numpy()
        else:
            batch_size, seq_len = input_ids.shape
            position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)
            for i in range(3):
                position_ids[i] = np.arange(seq_len).reshape(1, -1)
        
        # Attention mask
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
    
    def run_vision_encoder(self, pixel_values: np.ndarray, image_grid_thw: np.ndarray) -> np.ndarray:
        """
        Run vision encoder on preprocessed image patches.
        
        Args:
            pixel_values: [num_patches, features]
            image_grid_thw: [1, 3] or [[t, h, w]]
        
        Returns:
            vision_output: [num_vision_tokens, hidden_size]
        """
        # Compute position IDs for rotary embeddings
        pos_ids, max_grid_size = compute_pos_ids(image_grid_thw)
        
        vision_inputs = {
            "pixel_values": pixel_values.astype(np.float32),
            "pos_ids": pos_ids.numpy().astype(np.int64),
            "max_grid_size": np.array(max_grid_size, dtype=np.int64),
        }
        
        vision_session = self.session_mgr.get_session("vision")
        vision_output = vision_session.run(None, vision_inputs)[0]
        
        return vision_output
    
    def run_embedding(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Run embedding layer on token IDs.
        
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        embed_session = self.session_mgr.get_session("embedding")
        embeddings = embed_session.run(None, {"input_ids": input_ids.astype(np.int64)})[0]
        return embeddings
    
    def merge_embeddings(
        self, 
        token_embeddings: np.ndarray, 
        vision_output: np.ndarray, 
        input_ids: np.ndarray
    ) -> np.ndarray:
        """
        Merge token embeddings with vision embeddings.
        
        Replaces image token positions with vision embeddings.
        """
        inputs_embeds = token_embeddings.copy()
        image_mask = input_ids == self.image_token_id
        
        # Flatten vision output and insert at image token positions
        vision_flat = vision_output.reshape(-1, vision_output.shape[-1])
        inputs_embeds[image_mask] = vision_flat
        
        return inputs_embeds
    
    def run_prefill(
        self, 
        inputs_embeds: np.ndarray, 
        attention_mask: np.ndarray, 
        position_ids: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run LLM prefill phase.
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            kv_cache: dict with past_key_i, past_value_i for i in 0..15
        """
        prefill_inputs = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }
        
        prefill_session = self.session_mgr.get_session("prefill")
        prefill_outputs = prefill_session.run(None, prefill_inputs)
        
        logits = prefill_outputs[0]
        
        # Extract KV cache
        kv_cache = {}
        for i in range(NUM_LAYERS):
            kv_cache[f"past_key_{i}"] = prefill_outputs[1 + i * 2]
            kv_cache[f"past_value_{i}"] = prefill_outputs[2 + i * 2]
        
        return logits, kv_cache
    
    def run_decode_step(
        self,
        token_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        kv_cache: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run one decode step with KV cache.
        
        Returns:
            logits: [batch, 1, vocab_size]
            updated_kv_cache
        """
        decode_inputs = {
            "inputs_embeds": token_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
            **{k: v.astype(np.float32) for k, v in kv_cache.items()},
        }
        
        decode_session = self.session_mgr.get_session("decode")
        decode_outputs = decode_session.run(None, decode_inputs)
        
        logits = decode_outputs[0]
        
        # Extract updated KV cache
        new_kv_cache = {}
        for i in range(NUM_LAYERS):
            new_kv_cache[f"past_key_{i}"] = decode_outputs[1 + i * 2]
            new_kv_cache[f"past_value_{i}"] = decode_outputs[2 + i * 2]
        
        return logits, new_kv_cache
    
    def generate(
        self, 
        image: Image.Image, 
        prompt: str = "OCR:",
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[int], float, float]:
        """
        Full generation loop for a single image.
        
        Returns:
            generated_tokens: list of token IDs
            prefill_time_ms: time for prefill phase
            decode_time_ms: time for decode phase
        """
        max_new_tokens = max_new_tokens or self.max_tokens
        
        # 1. Preprocess
        inputs = self.preprocess_image(image, prompt)
        
        # 2. Vision encoding
        vision_output = self.run_vision_encoder(
            inputs["pixel_values"], 
            inputs["image_grid_thw"]
        )
        
        # 3. Token embedding
        token_embeddings = self.run_embedding(inputs["input_ids"])
        
        # 4. Merge embeddings
        inputs_embeds = self.merge_embeddings(
            token_embeddings, 
            vision_output, 
            inputs["input_ids"]
        )
        
        # 5. Prefill
        prefill_start = time.perf_counter()
        logits, kv_cache = self.run_prefill(
            inputs_embeds,
            inputs["attention_mask"],
            inputs["position_ids"],
        )
        prefill_time_ms = (time.perf_counter() - prefill_start) * 1000
        
        # 6. Decode loop
        decode_start = time.perf_counter()
        
        generated_tokens = []
        batch_size = inputs["input_ids"].shape[0]
        current_seq_len = inputs["input_ids"].shape[1]
        position_ids = inputs["position_ids"]
        
        # Get first token
        next_token = np.argmax(logits[:, -1:, :], axis=-1)
        
        for step in range(max_new_tokens):
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            
            if token_id == self.eos_token_id:
                break
            
            # Get embedding for next token
            next_embeds = self.run_embedding(next_token)
            
            # Update position IDs
            next_pos = position_ids[:, :, -1:] + 1 + step
            
            # Update attention mask (extend by 1)
            total_seq_len = current_seq_len + step + 1
            decode_attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
            
            # Run decode step
            logits, kv_cache = self.run_decode_step(
                next_embeds,
                decode_attention_mask,
                next_pos,
                kv_cache,
            )
            
            # Get next token
            next_token = np.argmax(logits[:, -1:, :], axis=-1)
        
        decode_time_ms = (time.perf_counter() - decode_start) * 1000
        
        return generated_tokens, prefill_time_ms, decode_time_ms


# ============================================================
# Batch Inference Runner
# ============================================================
class BatchInferenceRunner:
    """
    Runs batch inference with different batch sizes.
    
    Note: Due to variable image sizes producing different vision token counts,
    we process images sequentially through the vision encoder.
    For LLM inference, we can batch if sequences are padded to same length.
    
    Current implementation: Sequential processing with batch-level timing.
    """
    
    def __init__(
        self,
        session_manager: ONNXSessionManager,
        processor: AutoProcessor,
        image_token_id: int,
        eos_token_id: int,
        max_tokens: int = 256,
    ):
        self.engine = GLMOcrONNXInference(
            session_manager=session_manager,
            processor=processor,
            image_token_id=image_token_id,
            eos_token_id=eos_token_id,
            max_tokens=max_tokens,
        )
        self.processor = processor
    
    def run_batch(
        self, 
        image_paths: List[str], 
        batch_size: int,
        prompt: str = "OCR:",
        max_new_tokens: Optional[int] = None,
    ) -> BatchResult:
        """
        Run inference on a batch of images.
        
        Images are processed in groups of batch_size.
        """
        results = []
        batch_start = time.perf_counter()
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    
                    start_time = time.perf_counter()
                    generated_tokens, prefill_ms, decode_ms = self.engine.generate(
                        image, 
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )
                    total_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Decode tokens to text
                    generated_text = self.processor.tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )
                    
                    # Calculate tokens per second
                    num_tokens = len(generated_tokens)
                    tokens_per_sec = num_tokens / (total_ms / 1000) if total_ms > 0 else 0
                    
                    results.append(InferenceResult(
                        image_path=img_path,
                        generated_tokens=generated_tokens,
                        generated_text=generated_text,
                        prefill_time_ms=prefill_ms,
                        decode_time_ms=decode_ms,
                        total_time_ms=total_ms,
                        tokens_per_second=tokens_per_sec,
                    ))
                    
                except Exception as e:
                    print(f"  [ERROR] {img_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        batch_total_ms = (time.perf_counter() - batch_start) * 1000
        num_images = len(results)
        avg_time = batch_total_ms / num_images if num_images > 0 else 0
        throughput = num_images / (batch_total_ms / 1000) if batch_total_ms > 0 else 0
        
        return BatchResult(
            batch_size=batch_size,
            results=results,
            total_time_ms=batch_total_ms,
            avg_time_per_image_ms=avg_time,
            throughput_images_per_sec=throughput,
        )


# ============================================================
# Output Validation
# ============================================================
def validate_batch_outputs(
    batch_results: List[BatchResult],
    reference_batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Validate that outputs are consistent across different batch sizes.
    
    Compare outputs from different batch sizes against a reference (batch_size=1).
    """
    print("\n" + "=" * 60)
    print("Output Consistency Validation")
    print("=" * 60)
    
    # Find reference results (batch_size = reference_batch_size)
    reference = None
    for br in batch_results:
        if br.batch_size == reference_batch_size:
            reference = br
            break
    
    if reference is None:
        print(f"  [WARN] Reference batch size {reference_batch_size} not found")
        return {"valid": False, "reason": "No reference"}
    
    # Build reference dict: image_path -> generated_tokens
    ref_outputs = {r.image_path: r.generated_tokens for r in reference.results}
    
    validation_results = {
        "valid": True,
        "mismatches": [],
        "details": {},
    }
    
    for br in batch_results:
        if br.batch_size == reference_batch_size:
            continue
        
        batch_valid = True
        batch_mismatches = []
        
        for result in br.results:
            ref_tokens = ref_outputs.get(result.image_path)
            if ref_tokens is None:
                continue
            
            if result.generated_tokens != ref_tokens:
                batch_valid = False
                batch_mismatches.append({
                    "image": result.image_path,
                    "ref_tokens": ref_tokens[:20],  # First 20 tokens
                    "batch_tokens": result.generated_tokens[:20],
                })
        
        status = "[PASS]" if batch_valid else "[FAIL]"
        print(f"  Batch size {br.batch_size}: {status}")
        
        if not batch_valid:
            validation_results["valid"] = False
            validation_results["mismatches"].extend(batch_mismatches)
            for mm in batch_mismatches[:3]:  # Show first 3 mismatches
                print(f"    Mismatch: {os.path.basename(mm['image'])}")
                print(f"      ref: {mm['ref_tokens']}")
                print(f"      got: {mm['batch_tokens']}")
        
        validation_results["details"][br.batch_size] = {
            "valid": batch_valid,
            "num_mismatches": len(batch_mismatches),
        }
    
    return validation_results


# ============================================================
# Performance Summary
# ============================================================
def print_performance_summary(batch_results: List[BatchResult]):
    """Print a performance summary table."""
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    print(f"\n{'Batch Size':<12} {'Total (ms)':<12} {'Avg/Image':<12} {'Throughput':<12}")
    print("-" * 48)
    
    for br in sorted(batch_results, key=lambda x: x.batch_size):
        print(f"{br.batch_size:<12} {br.total_time_ms:<12.1f} "
              f"{br.avg_time_per_image_ms:<12.1f} {br.throughput_images_per_sec:<12.2f}")
    
    # Per-image breakdown for smallest batch
    if batch_results:
        smallest = min(batch_results, key=lambda x: x.batch_size)
        print(f"\n--- Detailed breakdown (batch_size={smallest.batch_size}) ---")
        print(f"{'Image':<30} {'Prefill':<10} {'Decode':<10} {'Total':<10} {'Tok/s':<8}")
        print("-" * 68)
        
        for r in smallest.results[:5]:  # First 5 images
            img_name = os.path.basename(r.image_path)[:28]
            print(f"{img_name:<30} {r.prefill_time_ms:<10.1f} {r.decode_time_ms:<10.1f} "
                  f"{r.total_time_ms:<10.1f} {r.tokens_per_second:<8.1f}")


# ============================================================
# Main
# ============================================================
def get_test_images(image_dir: Optional[str], max_images: int = 10) -> List[str]:
    """Get list of test images."""
    images = []
    
    if image_dir and os.path.isdir(image_dir):
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            images.extend(Path(image_dir).glob(ext))
        images = [str(p) for p in sorted(images)[:max_images]]
    
    # Add default test images if available
    default_images = [
        r"d:\tmp\table-2026-01-01-202211.png",
        r"d:\tmp\formula_2025-8-2_17-28-16.jpg",
    ]
    
    for img_path in default_images:
        if os.path.exists(img_path) and img_path not in images:
            images.append(img_path)
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference and validation for GLM-OCR ONNX models"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=r"d:\tmp",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="2",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per image",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="OCR:",
        help="Prompt to use for OCR",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=ONNX_DIR,
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to original model (for processor/tokenizer)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="ONNX Runtime execution provider",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated text for each image",
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    print("=" * 60)
    print("GLM-OCR ONNX Batch Inference")
    print("=" * 60)
    print(f"ONNX Directory: {args.onnx_dir}")
    print(f"Model Path: {args.model_path}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Provider: {args.provider}")
    print()
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Get model config for special token IDs (use AutoConfig to avoid loading full model)
    print("Loading model config...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    image_token_id = config.image_token_id
    # eos_token_id may not be in model config, get from tokenizer
    eos_token_id = getattr(config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = processor.tokenizer.eos_token_id
    print(f"  image_token_id: {image_token_id}")
    print(f"  eos_token_id: {eos_token_id}")
    print()
    
    # Load ONNX sessions
    session_manager = ONNXSessionManager(args.onnx_dir, provider=args.provider)
    
    # Get test images
    image_paths = get_test_images(args.image_dir, args.max_images)
    
    if not image_paths:
        print("No test images found!")
        print(f"  Provide --image-dir or ensure default images exist")
        sys.exit(1)
    
    print(f"Test images ({len(image_paths)}):")
    for p in image_paths:
        print(f"  - {p}")
    print()
    
    # Create batch runner
    runner = BatchInferenceRunner(
        session_manager=session_manager,
        processor=processor,
        image_token_id=image_token_id,
        eos_token_id=eos_token_id,
        max_tokens=args.max_tokens,
    )
    
    # Run inference with different batch sizes
    all_results = []
    
    for batch_size in batch_sizes:
        print("-" * 60)
        print(f"Running with batch_size={batch_size}")
        print("-" * 60)
        
        batch_result = runner.run_batch(
            image_paths=image_paths,
            batch_size=batch_size,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
        )
        all_results.append(batch_result)
        
        print(f"  Processed {len(batch_result.results)} images")
        print(f"  Total time: {batch_result.total_time_ms:.1f} ms")
        print(f"  Avg per image: {batch_result.avg_time_per_image_ms:.1f} ms")
        print(f"  Throughput: {batch_result.throughput_images_per_sec:.2f} images/sec")
        
        if args.verbose:
            print("\n  Generated outputs:")
            for r in batch_result.results:
                img_name = os.path.basename(r.image_path)
                text_preview = r.generated_text[:100].replace("\n", " ")
                print(f"    [{img_name}]: {text_preview}...")
        
        print()
    
    # Validate outputs across batch sizes
    validation = validate_batch_outputs(all_results, reference_batch_size=1)
    
    # Print performance summary
    print_performance_summary(all_results)
    
    # Final status
    print("\n" + "=" * 60)
    if validation["valid"]:
        print("✓ ALL VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED - outputs differ across batch sizes")
    print("=" * 60)
    
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
