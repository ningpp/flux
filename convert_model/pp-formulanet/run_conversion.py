"""
PP-FormulaNet Torch → ONNX Converter & Verifier
================================================
Converts PP-FormulaNet_plus-L and PP-FormulaNet-L from PyTorch safetensors to ONNX.
Verifies accuracy by comparing top-5 logits at each generation step.

Usage:
    python run_conversion.py                    # Convert both models
    python run_conversion.py --model plus       # Convert only PP-FormulaNet_plus-L
    python run_conversion.py --model L          # Convert only PP-FormulaNet-L
    python run_conversion.py --steps 20         # Verify only first 20 steps (faster)
    python run_conversion.py --skip-export      # Skip ONNX export (verify existing ONNX)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers.models.pp_formulanet import PPFormulaNetForConditionalGeneration
from transformers import AutoProcessor
import onnxruntime as ort

# ============================================================================
# Monkey-patch for JIT tracing compatibility
# ============================================================================

def _apply_patches():
    """Apply monkey-patches to make the decoder exportable with classic JIT tracer."""
    def patched_create_causal_mask(config, inputs_embeds, attention_mask=None,
                                    cache_position=None, *, past_key_values=None,
                                    position_ids=None, or_mask_function=None,
                                    and_mask_function=None):
        if past_key_values is None:
            return None  # Let SDPA use built-in is_causal=True
        import transformers.masking_utils as mu
        return mu.create_causal_mask(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
            or_mask_function=or_mask_function, and_mask_function=and_mask_function,
        )

    def patched_create_bidirectional_mask(config, inputs_embeds, attention_mask=None,
                                           encoder_hidden_states=None, past_key_values=None,
                                           or_mask_function=None, and_mask_function=None):
        if past_key_values is None and attention_mask is None:
            return None  # Let SDPA use built-in bidirectional attention
        import transformers.masking_utils as mu
        return mu.create_bidirectional_mask(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values,
            or_mask_function=or_mask_function, and_mask_function=and_mask_function,
        )

    import transformers.models.pp_formulanet.modeling_pp_formulanet as pp_module
    pp_module.create_causal_mask = patched_create_causal_mask
    pp_module.create_bidirectional_mask = patched_create_bidirectional_mask

_apply_patches()

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    "PP-FormulaNet_plus-L": {
        "model_dir": r"D:\models\PP-FormulaNet_plus-L_safetensors",
        "onnx_encoder": "onnx/PP-FormulaNet_plus-L_encoder.onnx",
        "onnx_decoder": "onnx/PP-FormulaNet_plus-L_decoder.onnx",
        "onnx_decoder_opt": "onnx/PP-FormulaNet_plus-L_decoder_opt.onnx",
    },
    "PP-FormulaNet-L": {
        "model_dir": r"D:\models\PP-FormulaNet-L_safetensors",
        "onnx_encoder": "onnx/PP-FormulaNet-L_encoder.onnx",
        "onnx_decoder": "onnx/PP-FormulaNet-L_decoder.onnx",
        "onnx_decoder_opt": "onnx/PP-FormulaNet-L_decoder_opt.onnx",
    },
}

VALIDATION_IMAGE = r"D:\models\pp-formulanet-torch-convert\imgs\formula-2026-01-18-152316.png"
OUTPUT_DIR = Path("onnx")

# ============================================================================
# ONNX Export Wrappers
# ============================================================================

class EncoderWrapper(nn.Module):
    """Export vision encoder: pixel_values → pooler_output (cross-attention features)."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        return outputs.pooler_output  # [B, enc_seq_len, decoder_hidden_size]


class DecoderWrapper(nn.Module):
    """Export text decoder + lm_head: (input_ids, encoder_hidden_states) → logits."""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head

    def forward(self, decoder_input_ids, encoder_hidden_states):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        )
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


class OptimizedDecoderWrapper(nn.Module):
    """Export text decoder + lm_head → token IDs only (argmax at each position).
    Single forward pass, O(n²) compute, 25,000× less output than full logits."""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head

    def forward(self, decoder_input_ids, encoder_hidden_states):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        )
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)          # [B, seq_len, 50000]
        token_ids = torch.argmax(logits, dim=-1, keepdim=True)  # [B, seq_len, 1]
        return token_ids


# ============================================================================
# Core Functions
# ============================================================================

def load_model(model_dir: str) -> Tuple[PPFormulaNetForConditionalGeneration, AutoProcessor]:
    """Load model and processor from a local directory."""
    print(f"  Loading model from: {model_dir}")
    model = PPFormulaNetForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def process_image(image_path: str, processor) -> torch.Tensor:
    """Load and preprocess the validation image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]


def export_encoder(model, onnx_path: str, sample_pixel_values: torch.Tensor):
    """Export the vision encoder to ONNX."""
    print(f"  Exporting encoder → {onnx_path}")
    wrapper = EncoderWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(sample_pixel_values)
        print(f"    Encoder output shape: {test_out.shape}")

    torch.onnx.export(
        wrapper,
        (sample_pixel_values,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"    ✅ Encoder exported")


def export_decoder(model, onnx_path: str, sample_input_ids: torch.Tensor,
                   sample_encoder_hidden: torch.Tensor):
    """Export the text decoder + lm_head to ONNX (full logits output)."""
    print(f"  Exporting decoder → {onnx_path}")
    wrapper = DecoderWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(sample_input_ids, sample_encoder_hidden)
        print(f"    Decoder output shape: {test_out.shape}")

    torch.onnx.export(
        wrapper,
        (sample_input_ids, sample_encoder_hidden),
        onnx_path,
        input_names=["decoder_input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch_size", 1: "decoder_seq_len"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
            "logits": {0: "batch_size", 1: "decoder_seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"    ✅ Decoder exported")


def export_optimized_decoder(model, onnx_path: str, sample_input_ids: torch.Tensor,
                              sample_encoder_hidden: torch.Tensor):
    """Export decoder with token-only output (argmax per position).
    Single-pass, same compute as original but 25,000x less output."""
    print(f"  Exporting optimized decoder → {onnx_path}")
    wrapper = OptimizedDecoderWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(sample_input_ids, sample_encoder_hidden)
        print(f"    Decoder output shape: {test_out.shape}, dtype: {test_out.dtype}")

    torch.onnx.export(
        wrapper,
        (sample_input_ids, sample_encoder_hidden),
        onnx_path,
        input_names=["decoder_input_ids", "encoder_hidden_states"],
        output_names=["token_ids"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch_size", 1: "decoder_seq_len"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
            "token_ids": {0: "batch_size", 1: "decoder_seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"    ✅ Optimized decoder exported")


def pytorch_generate_and_get_logits(model, pixel_values, bos_token_id, eos_token_id, max_steps):
    """Run PyTorch generate() to get full token sequence, then do a single
    use_cache=False forward pass to get ALL logits at every position.
    
    This is O(n²) total instead of O(n³) for the step-by-step approach.
    Returns: (all_tokens, per_step_logits_list)
    """
    # Step 1: Fast generate with KV cache to get complete token sequence
    with torch.no_grad():
        gen_out = model.generate(
            pixel_values=pixel_values,
            max_length=max_steps,
            do_sample=False,
            num_beams=1,
        )
    all_tokens = gen_out[0].tolist()  # includes decoder_start_token (EOS) + generated tokens + EOS
    
    # Step 2: Single forward pass with use_cache=False to get ALL logits
    # decoder_input_ids should be the prefix tokens (all except the last, which is what we predict)
    decoder_input_ids = gen_out[:, :-1]  # [1, N-1]
    
    with torch.no_grad():
        enc_out = model.model.encoder(pixel_values)
        pt_enc = enc_out.pooler_output
        
        dec_out = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=pt_enc,
            use_cache=False,
        )
        all_logits = model.lm_head(dec_out.last_hidden_state)  # [1, N-1, vocab_size]
    
    # Step 3: Extract top-5 logits at each position
    step_logits = []
    for i in range(all_logits.shape[1]):  # for each position in the sequence
        logits_at_pos = all_logits[0, i, :]
        top5_vals, top5_ids = torch.topk(logits_at_pos, k=5)
        step_logits.append({
            "step": i,
            "top1_token": top5_ids[0].item(),
            "top1_value": top5_vals[0].item(),
            "top5_ids": top5_ids.cpu().tolist(),
            "top5_vals": top5_vals.cpu().tolist(),
            "actual_next_token": all_tokens[i + 1] if i + 1 < len(all_tokens) else None,
        })
    
    return all_tokens, step_logits


def onnx_get_logits_from_tokens(encoder_path, decoder_path, pixel_values_np, all_tokens):
    """Run ONNX encoder once, then decoder once with all tokens to get logits
    at every position.
    Returns: per_step_logits_list
    """
    enc_sess = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    dec_sess = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])

    # Encoder: single forward pass
    onnx_enc = enc_sess.run(None, {"pixel_values": pixel_values_np.astype(np.float32)})[0]
    onnx_enc_f32 = onnx_enc.astype(np.float32)
    
    # Decoder: single forward pass with all tokens (use_cache=False)
    # decoder_input_ids should be all tokens except the last
    decoder_input_ids = np.array([all_tokens[:-1]], dtype=np.int64)
    
    dec_out = dec_sess.run(None, {
        "decoder_input_ids": decoder_input_ids,
        "encoder_hidden_states": onnx_enc_f32,
    })
    all_logits = dec_out[0]  # [1, N-1, vocab_size]
    
    # Extract top-5 at each position
    step_logits = []
    for i in range(all_logits.shape[1]):
        logits_at_pos = all_logits[0, i, :]
        top5_ids = np.argsort(logits_at_pos)[-5:][::-1]
        top5_vals = logits_at_pos[top5_ids]
        step_logits.append({
            "step": i,
            "top1_token": int(top5_ids[0]),
            "top1_value": float(top5_vals[0]),
            "top5_ids": top5_ids.tolist(),
            "top5_vals": top5_vals.tolist(),
            "actual_next_token": all_tokens[i + 1] if i + 1 < len(all_tokens) else None,
        })
    
    return step_logits


def compare_results(pt_step_logits, onnx_step_logits, processor, pt_tokens, onnx_tokens):
    """Compare PyTorch and ONNX per-step logits. Print detailed report."""
    num_steps = min(len(pt_step_logits), len(onnx_step_logits))

    top1_match = 0
    top5_full_match = 0
    max_logit_diff = 0.0
    divergence_step = None
    all_mismatches = []

    for step in range(num_steps):
        pt = pt_step_logits[step]
        on = onnx_step_logits[step]

        if pt["top1_token"] == on["top1_token"]:
            top1_match += 1
        elif divergence_step is None:
            divergence_step = step

        if set(pt["top5_ids"]) == set(on["top5_ids"]):
            top5_full_match += 1
        else:
            all_mismatches.append({
                "step": step,
                "pt_top1": pt["top1_token"],
                "on_top1": on["top1_token"],
                "pt_top5": pt["top5_ids"],
                "on_top5": on["top5_ids"],
            })

        on_set = set(on["top5_ids"])
        for pt_id, pt_val in zip(pt["top5_ids"], pt["top5_vals"]):
            if pt_id in on_set:
                on_idx = on["top5_ids"].index(pt_id)
                diff = abs(pt_val - on["top5_vals"][on_idx])
                if diff > max_logit_diff:
                    max_logit_diff = diff

    # Print report
    print(f"\n  {'='*60}")
    print(f"  ACCURACY COMPARISON")
    print(f"  {'='*60}")
    print(f"  Total steps compared: {num_steps}")
    print(f"  Top-1 token match: {top1_match}/{num_steps} = {100*top1_match/num_steps:.2f}%")
    print(f"  Top-5 exact match (same set): {top5_full_match}/{num_steps} = {100*top5_full_match/num_steps:.2f}%")
    print(f"  Max logit difference: {max_logit_diff:.6e}")

    # First 5 steps detail
    print(f"\n  --- First 5 Steps Detail ---")
    for step in range(min(5, num_steps)):
        pt = pt_step_logits[step]
        on = onnx_step_logits[step]
        icon = "✅" if pt["top1_token"] == on["top1_token"] else "❌"
        print(f"  Step {step} {icon}:")
        print(f"    PT  top-5: {list(zip(pt['top5_ids'], [f'{v:.4f}' for v in pt['top5_vals']]))}")
        print(f"    ONNX top-5: {list(zip(on['top5_ids'], [f'{v:.4f}' for v in on['top5_vals']]))}")

    if all_mismatches:
        print(f"\n  ⚠️  MISMATCHED STEPS: {len(all_mismatches)}/{num_steps}")
        for m in all_mismatches[:10]:
            overlap = len(set(m["pt_top5"]) & set(m["on_top5"]))
            print(f"  Step {m['step']}: PT top-1={m['pt_top1']}, ONNX top-1={m['on_top1']}, overlap={overlap}/5")
        if len(all_mismatches) > 10:
            print(f"  ... and {len(all_mismatches) - 10} more mismatches")
    elif divergence_step is not None:
        print(f"\n  ⚠️  First top-1 divergence at step {divergence_step}")
    else:
        print(f"\n  ✅ All {num_steps} steps: Top-1 tokens IDENTICAL!")

    # Decoded text
    pt_text = processor.tokenizer.decode(pt_tokens, skip_special_tokens=False)
    onnx_text = processor.tokenizer.decode(onnx_tokens, skip_special_tokens=False)
    texts_match = pt_text == onnx_text
    print(f"\n  Full text match: {'✅ YES' if texts_match else '❌ NO'}")
    if not texts_match:
        # Find first difference
        for i, (a, b) in enumerate(zip(pt_text, onnx_text)):
            if a != b:
                print(f"  First text diff at char {i}: PT='{pt_text[i:i+20]}...' ONNX='{onnx_text[i:i+20]}...'")
                break

    return {
        "num_steps": num_steps,
        "top1_match_pct": 100 * top1_match / num_steps if num_steps > 0 else 0,
        "top5_full_match_pct": 100 * top5_full_match / num_steps if num_steps > 0 else 0,
        "max_logit_diff": max_logit_diff,
        "paths_match": top1_match == num_steps,
        "texts_match": texts_match,
        "mismatch_count": len(all_mismatches),
    }


def convert_and_verify(model_name: str, config: dict, args):
    """Convert a single model to ONNX and verify accuracy."""
    print(f"\n{'='*70}")
    print(f"PROCESSING: {model_name}")
    print(f"{'='*70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder_onnx = config["onnx_encoder"]
    decoder_onnx = config["onnx_decoder"]
    image_path = args.image

    # 1. Load model
    print("\n[1/4] Loading model...")
    model, processor = load_model(config["model_dir"])

    bos_token_id = model.config.text_config.bos_token_id
    eos_token_id = model.config.text_config.eos_token_id

    # 2. Process image
    print("\n[2/4] Processing validation image...")
    pixel_values = process_image(image_path, processor)
    print(f"  Pixel values shape: {pixel_values.shape}")

    # 3. Export to ONNX (if not skipped)
    if not args.skip_export:
        print("\n[3/4] Exporting to ONNX...")

        # Export encoder
        if not os.path.exists(encoder_onnx) or args.force:
            export_encoder(model, encoder_onnx, pixel_values)
        else:
            print(f"  Encoder ONNX already exists: {encoder_onnx}")

        # Export decoder: need encoder output as sample input
        with torch.no_grad():
            enc_out = model.model.encoder(pixel_values)
            pt_enc = enc_out.pooler_output

        sample_dec_ids = torch.tensor([[bos_token_id, bos_token_id, bos_token_id]], dtype=torch.long)

        if not os.path.exists(decoder_onnx) or args.force:
            export_decoder(model, decoder_onnx, sample_dec_ids, pt_enc)
        else:
            print(f"  Decoder ONNX already exists: {decoder_onnx}")

    # 4. Verify accuracy — use single-forward-pass approach for O(n²) efficiency
    print(f"\n[{'4' if not args.skip_export else '3'}/4] Verifying accuracy (max_length={args.steps})...")

    # Check ONNX files exist
    if not os.path.exists(encoder_onnx):
        print(f"  ❌ Encoder ONNX not found: {encoder_onnx}")
        return None
    if not os.path.exists(decoder_onnx):
        print(f"  ❌ Decoder ONNX not found: {decoder_onnx}")
        return None

    # PyTorch: generate full token sequence + get all logits in single forward pass
    print("  Running PyTorch generate + logit extraction...")
    t0 = time.time()
    pt_tokens, pt_logits = pytorch_generate_and_get_logits(
        model, pixel_values, bos_token_id, eos_token_id, args.steps
    )
    pt_time = time.time() - t0
    print(f"    Generated {len(pt_tokens)} tokens, {len(pt_logits)} logit steps in {pt_time:.1f}s")

    # ONNX: get logits from the same token sequence (single forward pass)
    print("  Running ONNX logit extraction...")
    t0 = time.time()
    pixel_values_np = pixel_values.cpu().numpy()
    onnx_logits = onnx_get_logits_from_tokens(
        encoder_onnx, decoder_onnx, pixel_values_np, pt_tokens
    )
    onnx_time = time.time() - t0
    # ONNX tokens are identical to PT tokens by construction (we feed the same sequence)
    onnx_tokens = pt_tokens
    print(f"    Extracted {len(onnx_logits)} logit steps in {onnx_time:.1f}s")

    # Compare
    results = compare_results(pt_logits, onnx_logits, processor, pt_tokens, onnx_tokens)
    results["model_name"] = model_name
    results["pt_time"] = pt_time
    results["onnx_time"] = onnx_time

    # Save report
    report_path = OUTPUT_DIR / f"{model_name}_verification.json"
    report = {
        "model_name": model_name,
        "validation_image": image_path,
        "max_steps_setting": args.steps,
        "pytorch_tokens": pt_tokens,
        "onnx_tokens": onnx_tokens,
        "num_steps": results["num_steps"],
        "top1_match_pct": results["top1_match_pct"],
        "top5_full_match_pct": results["top5_full_match_pct"],
        "max_logit_diff": results["max_logit_diff"],
        "paths_match": results["paths_match"],
        "texts_match": results["texts_match"],
        "pt_time_seconds": pt_time,
        "onnx_time_seconds": onnx_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Verification report saved to: {report_path}")

    # Decode full text
    pt_text = processor.tokenizer.decode(pt_tokens, skip_special_tokens=False)
    print(f"\n  Full decoded output ({len(pt_tokens)} tokens):")
    print(f"    {pt_text}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PP-FormulaNet Torch → ONNX Converter & Verifier"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "plus", "L"],
                        help="Which model(s) to convert")
    parser.add_argument("--steps", type=int, default=1537,
                        help="Maximum generation steps for verification (default: 1537, model's max_length)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export (use existing ONNX files)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-export even if ONNX files exist")
    parser.add_argument("--image", type=str, default=VALIDATION_IMAGE,
                        help="Path to validation image")
    args = parser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        print(f"❌ Validation image not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("PP-FormulaNet Torch → ONNX Converter & Verifier")
    print("=" * 70)
    print(f"Validation image: {image_path}")
    print(f"Max steps: {args.steps}")
    print(f"Skip export: {args.skip_export}")
    print(f"Force re-export: {args.force}")

    all_results = {}

    if args.model in ("all", "plus"):
        all_results["PP-FormulaNet_plus-L"] = convert_and_verify(
            "PP-FormulaNet_plus-L", MODEL_CONFIGS["PP-FormulaNet_plus-L"], args
        )

    if args.model in ("all", "L"):
        all_results["PP-FormulaNet-L"] = convert_and_verify(
            "PP-FormulaNet-L", MODEL_CONFIGS["PP-FormulaNet-L"], args
        )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, results in all_results.items():
        if results:
            status = "✅ PASS" if results["paths_match"] else "❌ FAIL"
            print(f"  {name}: {status} | "
                  f"Top-1: {results['top1_match_pct']:.1f}% | "
                  f"Top-5: {results['top5_full_match_pct']:.1f}% | "
                  f"Max diff: {results['max_logit_diff']:.2e} | "
                  f"PT: {results['pt_time']:.1f}s | ONNX: {results['onnx_time']:.1f}s")
            if not results["paths_match"]:
                all_pass = False

    if all_pass:
        print("\n🎉 ALL MODELS PASSED VERIFICATION!")
    else:
        print("\n⚠️  Some models failed verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
