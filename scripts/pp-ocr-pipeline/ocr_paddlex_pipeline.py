"""
PaddleX OCR Pipeline - 使用PaddleX官方API运行OCR推理，输出所有中间结果。
用于与Java端OCRPipeline对比验证。

Usage (CPU):
  conda run -n paddlex python scripts/pp-ocr-pipeline/ocr_paddlex_pipeline.py --device cpu
Usage (GPU):
  conda run -n paddlex python scripts/pp-ocr-pipeline/ocr_paddlex_pipeline.py --device gpu
"""

import argparse
import json
import os
import numpy as np
from paddleocr import PaddleOCR


MODEL_ROOT = r"D:\models"

MODEL_DIRS = {
    "doc_ori": os.path.join(MODEL_ROOT, "PP-LCNet_x1_0_doc_ori_onnx"),
    "textline_ori": os.path.join(MODEL_ROOT, "PP-LCNet_x1_0_textline_ori_onnx"),
    "det": os.path.join(MODEL_ROOT, "PP-OCRv6_medium_det_onnx"),
    "rec": os.path.join(MODEL_ROOT, "PP-OCRv6_medium_rec_onnx"),
}


def run_pipeline(image_path, device="cpu"):
    """Run PaddleX OCR pipeline and return all intermediate results."""
    device_str = device if device == "cpu" else "gpu:0"

    print(f"\n{'='*60}")
    print(f"PaddleX OCR Pipeline on {device.upper()}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")

    ocr = PaddleOCR(
        engine="onnxruntime",
        device=device_str,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        doc_orientation_classify_model_dir=MODEL_DIRS["doc_ori"],
        textline_orientation_model_dir=MODEL_DIRS["textline_ori"],
        text_detection_model_dir=MODEL_DIRS["det"],
        text_recognition_model_dir=MODEL_DIRS["rec"],
    )

    results = ocr.predict(image_path)
    res = results[0]

    output = {
        "device": device,
        "image": os.path.abspath(image_path),
        "model_settings": res.get("model_settings", {}),
        "text_det_params": res.get("text_det_params", {}),
        "text_type": res.get("text_type", ""),
    }

    # ── Doc Orientation ──
    doc_pre_res = res.get("doc_preprocessor_res")
    if doc_pre_res is not None:
        doc_angle = doc_pre_res.get("angle", 0)
        doc_model_settings = doc_pre_res.get("model_settings", {})
        output["doc_orientation"] = {
            "angle": doc_angle,
            "model_settings": doc_model_settings,
        }
        print(f"\n[Doc Orientation] angle={doc_angle}°")
    else:
        output["doc_orientation"] = {"angle": 0, "model_settings": {}}
        print(f"\n[Doc Orientation] not available, angle=0°")

    # ── Text Detection ──
    dt_polys = res.get("dt_polys", [])
    output["detection"] = {
        "num_regions": len(dt_polys),
        "polygons": [poly.tolist() for poly in dt_polys],
    }
    print(f"\n[Text Detection] Found {len(dt_polys)} text regions")
    for i, poly in enumerate(dt_polys):
        print(f"  Poly {i}: {poly.tolist()}")

    # ── Text Line Orientation ──
    textline_angles = res.get("textline_orientation_angles", [])
    output["textline_orientation"] = {
        "angles": textline_angles,
    }
    print(f"\n[Text Line Orientation]")
    for i, angle in enumerate(textline_angles):
        label = "180_degree" if angle == 1 else "0_degree"
        print(f"  Line {i}: angle_indicator={angle}, label={label}")

    # ── Text Recognition ──
    rec_texts = res.get("rec_texts", [])
    rec_scores = res.get("rec_scores", [])
    rec_polys = res.get("rec_polys", [])

    output["recognition"] = []
    print(f"\n[Text Recognition]")
    for i in range(len(rec_texts)):
        text = rec_texts[i]
        score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        poly = rec_polys[i].tolist() if i < len(rec_polys) else None
        ori_angle = textline_angles[i] if i < len(textline_angles) else -1
        ori_label = "180_degree" if ori_angle == 1 else "0_degree" if ori_angle == 0 else "unknown"

        rec_item = {
            "index": i,
            "text": text,
            "confidence": round(score, 4),
            "textline_orientation": ori_label,
            "polygon": poly,
        }
        output["recognition"].append(rec_item)
        print(f"  Line {i}: text=\"{text}\", conf={score:.4f}, ori={ori_label}")

    return output


def main():
    parser = argparse.ArgumentParser(description="PaddleX OCR Pipeline - Full intermediate results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Device: cpu or gpu")
    parser.add_argument("--image", type=str,
                        default=r"scripts\pp-ocr-pipeline\ocr-pipeline-2026-06-12-211832.png",
                        help="Path to input image")
    args = parser.parse_args()

    # Run on CPU
    cpu_output = run_pipeline(args.image, device="cpu")

    # Run on GPU if requested
    gpu_output = None
    if args.device == "gpu":
        gpu_output = run_pipeline(args.image, device="gpu")

    # Save results
    result = {"cpu": cpu_output}
    if gpu_output:
        result["gpu"] = gpu_output

    output_path = os.path.join(os.path.dirname(args.image), "ocr_paddlex_pipeline_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Compare CPU vs GPU
    if gpu_output:
        print(f"\n{'='*60}")
        print("CPU vs GPU Comparison:")
        print(f"{'='*60}")

        # Doc orientation
        cpu_angle = cpu_output["doc_orientation"]["angle"]
        gpu_angle = gpu_output["doc_orientation"]["angle"]
        match = "MATCH" if cpu_angle == gpu_angle else "DIFF"
        print(f"  Doc Orientation: CPU={cpu_angle}° GPU={gpu_angle}° [{match}]")

        # Detection count
        cpu_det = cpu_output["detection"]["num_regions"]
        gpu_det = gpu_output["detection"]["num_regions"]
        match = "MATCH" if cpu_det == gpu_det else "DIFF"
        print(f"  Detection count: CPU={cpu_det} GPU={gpu_det} [{match}]")

        # Text line orientation
        cpu_angles = cpu_output["textline_orientation"]["angles"]
        gpu_angles = gpu_output["textline_orientation"]["angles"]
        match = "MATCH" if cpu_angles == gpu_angles else "DIFF"
        print(f"  TextLine Orientation: CPU={cpu_angles} GPU={gpu_angles} [{match}]")

        # Recognition
        for cpu_r, gpu_r in zip(cpu_output["recognition"], gpu_output["recognition"]):
            match = "MATCH" if cpu_r["text"] == gpu_r["text"] else "DIFF"
            print(f"  Line {cpu_r['index']}: CPU=\"{cpu_r['text']}\" GPU=\"{gpu_r['text']}\" [{match}]")


if __name__ == "__main__":
    main()
