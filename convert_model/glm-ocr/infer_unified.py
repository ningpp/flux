"""
GLM-OCR Hybrid Model Inference Script

This script performs OCR on images using GLM-OCR ONNX models.
Uses llm_prefill.onnx for prefill phase (variable seq_len) and
llm_unified.onnx for decode phase (seq_len=1).

Usage:
    python infer_unified.py --image D:\tmp\formula-2026028-105537.jpg
"""

import os
import sys
import time
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = r"D:\models\GLM-OCR"
ONNX_DIR = r"D:\models\onnx\GLM-OCR"
UNIFIED_MODEL_NAME = "llm_unified.onnx"  # Or "llm_unified_fp16.onnx"

NUM_LAYERS = 16
HIDDEN_SIZE = 1536
NUM_KV_HEADS = 8  # From config.num_key_value_heads
HEAD_DIM = 128    # From config.head_dim
SPATIAL_MERGE_SIZE = 2


# ============================================================
# Helper Functions
# ============================================================
def compute_pos_ids(grid_thw, spatial_merge_size=SPATIAL_MERGE_SIZE):
    """
    Pre-compute position IDs for rotary embeddings.
    """
    import torch

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
# GLM-OCR Unified Inference Engine
# ============================================================
class GLMOcrHybridInference:
    """
    GLM-OCR inference engine using hybrid ONNX models.

    Uses two models:
    - llm_prefill.onnx: For prefill phase (handles variable seq_len)
    - llm_unified.onnx: For decode phase (seq_len=1)
    """

    def __init__(
            self,
            onnx_dir: str,
            model_path: str,
            unified_model_name: str = UNIFIED_MODEL_NAME,
            provider: str = "CPUExecutionProvider",
    ):
        print("Loading GLM-OCR Hybrid Models...")

        # Load processor
        print("  Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load config for special token IDs
        print("  Loading config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.image_token_id = config.image_token_id
        self.eos_token_id = getattr(config, "eos_token_id", None)
        if self.eos_token_id is None:
            self.eos_token_id = self.processor.tokenizer.eos_token_id

        print(f"    image_token_id: {self.image_token_id}")
        print(f"    eos_token_id: {self.eos_token_id}")

        # Load ONNX sessions
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Vision encoder
        vision_path = os.path.join(onnx_dir, "vision_encoder.onnx")
        if not os.path.exists(vision_path):
            raise FileNotFoundError(f"Vision encoder not found: {vision_path}")
        self.vision_session = ort.InferenceSession(vision_path, opts, providers=[provider])
        print(f"  ✓ Loaded vision_encoder.onnx")

        # Embedding
        embed_path = os.path.join(onnx_dir, "embedding.onnx")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Embedding model not found: {embed_path}")
        self.embed_session = ort.InferenceSession(embed_path, opts, providers=[provider])
        print(f"  ✓ Loaded embedding.onnx")

        # Prefill LLM (for variable seq_len)
        prefill_path = os.path.join(onnx_dir, "llm_prefill.onnx")
        if not os.path.exists(prefill_path):
            raise FileNotFoundError(f"Prefill model not found: {prefill_path}")
        self.prefill_session = ort.InferenceSession(prefill_path, opts, providers=[provider])
        print(f"  ✓ Loaded llm_prefill.onnx")

        # Unified LLM (for decode phase, seq_len=1)
        unified_path = os.path.join(onnx_dir, unified_model_name)
        if not os.path.exists(unified_path):
            raise FileNotFoundError(f"Unified model not found: {unified_path}")
        self.unified_session = ort.InferenceSession(unified_path, opts, providers=[provider])
        print(f"  ✓ Loaded {unified_model_name}")
        print()

        # Model config
        self.num_layers = NUM_LAYERS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM

    def preprocess_image(
            self,
            image: Image.Image,
            prompt: str = "OCR:"
    ) -> Dict[str, np.ndarray]:
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
        # Convert image_grid_thw to numpy, handling various input types
        if isinstance(inputs["image_grid_thw"], np.ndarray):
            image_grid_thw = inputs["image_grid_thw"]
        elif isinstance(inputs["image_grid_thw"], (list, tuple)):
            image_grid_thw = np.asarray(inputs["image_grid_thw"], dtype=np.int64)
        else:
            # For PyTorch tensors or other types, first convert to numpy without copy keyword
            # Then convert dtype if needed
            temp = np.asarray(inputs["image_grid_thw"])
            image_grid_thw = temp.astype(np.int64, copy=False)

        # Get or create position_ids
        if "position_ids" in inputs and inputs["position_ids"] is not None:
            position_ids = inputs["position_ids"].numpy()
        else:
            # Compute 3D position_ids matching transformers model.generate():
            #  - Text tokens: all 3 dims get the same sequential position
            #  - Vision tokens: dim0 = temporal (constant per frame),
            #    dim1 = spatial row, dim2 = spatial column
            #    (all offset by number of preceding text positions)
            #  - Text after vision: continues from max(vision positions) + 1
            batch_size, seq_len = input_ids.shape
            position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

            for b in range(batch_size):
                ids = input_ids[b]
                image_mask = (ids == self.image_token_id)
                pos = 0
                i = 0
                img_idx = 0

                while i < seq_len:
                    if not image_mask[i]:
                        # Text token — all 3 dims share the same position
                        position_ids[0, b, i] = pos
                        position_ids[1, b, i] = pos
                        position_ids[2, b, i] = pos
                        pos += 1
                        i += 1
                    else:
                        # Vision token region
                        t_grid = int(image_grid_thw[img_idx, 0])
                        h_grid = int(image_grid_thw[img_idx, 1])
                        w_grid = int(image_grid_thw[img_idx, 2])
                        merged_h = h_grid // SPATIAL_MERGE_SIZE
                        merged_w = w_grid // SPATIAL_MERGE_SIZE
                        num_vision_tokens = t_grid * merged_h * merged_w

                        temporal_pos = pos
                        for vi in range(num_vision_tokens):
                            row = (vi % (merged_h * merged_w)) // merged_w
                            col = (vi % (merged_h * merged_w)) % merged_w
                            position_ids[0, b, i + vi] = temporal_pos
                            position_ids[1, b, i + vi] = pos + row
                            position_ids[2, b, i + vi] = pos + col

                        pos = pos + max(merged_h, merged_w)
                        i += num_vision_tokens
                        img_idx += 1

        # Attention mask
        attention_mask = np.ones_like(input_ids, dtype=np.int64)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def run_vision_encoder(
            self,
            pixel_values: np.ndarray,
            image_grid_thw: np.ndarray
    ) -> np.ndarray:
        """
        Run vision encoder on preprocessed image patches.

        Args:
            pixel_values: [num_patches, features]
            image_grid_thw: [1, 3] or [[t, h, w]]

        Returns:
            vision_output: [num_vision_tokens, hidden_size]
        """
        # Compute position IDs for rotary embeddings
        import torch
        pos_ids, max_grid_size = compute_pos_ids(image_grid_thw)

        vision_inputs = {
            "pixel_values": pixel_values.astype(np.float32),
            "pos_ids": pos_ids.numpy().astype(np.int64),
            "max_grid_size": np.array(max_grid_size, dtype=np.int64),
        }

        vision_output = self.vision_session.run(None, vision_inputs)[0]

        return vision_output

    def run_embedding(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Run embedding layer on token IDs.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        embeddings = self.embed_session.run(
            None,
            {"input_ids": input_ids.astype(np.int64)}
        )[0]
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

    def create_empty_kv_cache(self, batch_size: int = 1) -> List[np.ndarray]:
        """
        Create empty KV cache tensors for prefill phase.

        Returns list of 32 tensors (16 layers x 2) with shape [batch, heads, 0, head_dim]
        """
        kv_cache = []
        for _ in range(self.num_layers):
            empty_key = np.zeros((batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
            empty_value = np.zeros((batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32)
            kv_cache.extend([empty_key, empty_value])
        return kv_cache

    def run_prefill_model(
            self,
            inputs_embeds: np.ndarray,
            attention_mask: np.ndarray,
            position_ids: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run the prefill LLM model (handles variable seq_len).

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            position_ids: [3, batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
            kv_cache: List of 32 tensors (16 layers x 2)
        """
        # Build input dict for prefill model (no KV cache inputs)
        prefill_inputs = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }

        # Run inference
        outputs = self.prefill_session.run(None, prefill_inputs)

        # Extract logits and KV cache
        logits = outputs[0]
        kv_cache = []
        for i in range(self.num_layers):
            kv_cache.append(outputs[1 + i * 2])
            kv_cache.append(outputs[2 + i * 2])

        return logits, kv_cache

    def run_decode_model(
            self,
            inputs_embeds: np.ndarray,
            attention_mask: np.ndarray,
            position_ids: np.ndarray,
            kv_cache: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run the unified LLM model for decode (seq_len=1).

        Args:
            inputs_embeds: [batch, 1, hidden_size]
            attention_mask: [batch, total_seq_len]
            position_ids: [3, batch, 1]
            kv_cache: List of 32 tensors (16 layers x 2) with real cache

        Returns:
            logits: [batch, 1, vocab_size]
            new_kv_cache: List of 32 tensors (16 layers x 2)
        """
        # Build input dict
        decode_inputs = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }

        # Add KV cache tensors
        for i in range(self.num_layers):
            decode_inputs[f"past_key_{i}"] = kv_cache[i * 2].astype(np.float32)
            decode_inputs[f"past_value_{i}"] = kv_cache[i * 2 + 1].astype(np.float32)

        # Run inference
        outputs = self.unified_session.run(None, decode_inputs)

        # Extract logits and KV cache
        logits = outputs[0]
        new_kv_cache = []
        for i in range(self.num_layers):
            new_kv_cache.append(outputs[1 + i * 2])
            new_kv_cache.append(outputs[2 + i * 2])

        return logits, new_kv_cache

    def generate(
            self,
            image: Image.Image,
            prompt: str = "OCR:",
            max_new_tokens: int = 256,
            temperature: float = 1.0,
    ) -> str:
        """
        Full generation loop for a single image.

        Args:
            image: PIL Image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = greedy)

        Returns:
            generated_text: The generated text
        """
        print(f"Generating with max_tokens={max_new_tokens}, temperature={temperature}...")

        # 1. Preprocess
        inputs = self.preprocess_image(image, prompt)
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]

        print(f"  Input seq_len: {seq_len}")

        # 2. Vision encoding
        vision_output = self.run_vision_encoder(
            inputs["pixel_values"],
            inputs["image_grid_thw"]
        )
        print(f"  Vision tokens: {vision_output.shape[0]}")

        # 3. Token embedding
        token_embeddings = self.run_embedding(inputs["input_ids"])

        # 4. Merge embeddings
        inputs_embeds = self.merge_embeddings(
            token_embeddings,
            vision_output,
            inputs["input_ids"]
        )

        # 5. Prefill phase (using prefill model for variable seq_len)
        prefill_start = time.perf_counter()
        logits, kv_cache = self.run_prefill_model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
        )
        prefill_time = (time.perf_counter() - prefill_start) * 1000
        print(f"  Prefill time: {prefill_time:.1f} ms")

        # 6. Decode loop (using unified model for seq_len=1)
        decode_start = time.perf_counter()
        generated_tokens = []
        current_pos = inputs["position_ids"][:, :, -1:] + 1

        for step in range(max_new_tokens):
            # Sample next token
            if temperature == 1.0:
                next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            else:
                # Apply temperature sampling
                logits_temp = logits[:, -1, :] / temperature
                probs = np.exp(logits_temp - np.max(logits_temp, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                next_token = int(np.random.choice(logits.shape[-1], p=probs[0]))

            generated_tokens.append(next_token)

            if next_token == self.eos_token_id or next_token == 59246 or next_token == 59253:
                print(f"  EOS at step {step + 1}")
                break

            # Get embedding for next token
            next_token_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self.run_embedding(next_token_ids)

            # Update position IDs
            next_pos = current_pos + step

            # Update attention mask (extend by 1)
            total_seq_len = seq_len + step + 1
            decode_attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)

            # Run decode step
            logits, kv_cache = self.run_decode_model(
                inputs_embeds=next_embeds,
                attention_mask=decode_attention_mask,
                position_ids=next_pos,
                kv_cache=kv_cache,
            )

        decode_time = (time.perf_counter() - decode_start) * 1000
        print(f"  Decode time: {decode_time:.1f} ms ({len(generated_tokens)} tokens)")
        print(f"  Total time: {prefill_time + decode_time:.1f} ms")

        # Decode tokens to text
        generated_text = self.processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print("generated_tokens:")
        print(generated_tokens)
        print("-"*31)
        return generated_text


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Hybrid Model Inference (Prefill + Decode)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=r"D:\tmp\formula-2026028-105537.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=UNIFIED_MODEL_NAME,
        choices=["llm_unified.onnx", "llm_unified_fp16.onnx"],
        help="Unified model filename",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Formula Recognition:",
        help="Text prompt for OCR",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = greedy)",
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

    args = parser.parse_args()

    print("=" * 60)
    print("GLM-OCR Hybrid Model Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Model: {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Provider: {args.provider}")
    print()

    # Check image exists
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"  Image size: {image.size}")
    print()

    # Initialize inference engine
    engine = GLMOcrHybridInference(
        onnx_dir=args.onnx_dir,
        model_path=args.model_path,
        unified_model_name=args.model_name,
        provider=args.provider,
    )

    # Run inference
    print("-" * 60)
    start_time = time.perf_counter()
    result = engine.generate(
        image=image,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    total_time = (time.perf_counter() - start_time) * 1000

    # Print result
    print("-" * 60)
    print("\nGenerated Text:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"\nTotal inference time: {total_time:.1f} ms")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
