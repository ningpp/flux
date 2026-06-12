"""
对比PaddleX Python OCR Pipeline和Java OCR Pipeline的所有中间结果。

Usage:
  conda run -n paddlex python scripts/pp-ocr-pipeline/compare_python_java.py
"""

import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_RESULT = os.path.join(SCRIPT_DIR, "ocr_paddlex_pipeline_result.json")
JAVA_CPU_RESULT = os.path.join(SCRIPT_DIR, "ocr_pipeline_java_cpu_result.json")
JAVA_GPU_RESULT = os.path.join(SCRIPT_DIR, "ocr_pipeline_java_gpu_result.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_doc_orientation(py_data, java_data, label):
    """对比文档方向分类结果"""
    py_angle = py_data.get("doc_orientation", {}).get("angle", 0)
    java_label = java_data.get("doc_orientation", {}).get("label", "0")
    java_score = java_data.get("doc_orientation", {}).get("score", 0)

    # Java label是字符串如"0","90","180","270"，对应Python的angle
    py_angle_str = str(py_angle)
    match = "MATCH" if py_angle_str == java_label else "DIFF"

    print(f"\n  [{label}] Doc Orientation:")
    print(f"    Python: angle={py_angle}°")
    print(f"    Java:   label={java_label}, score={java_score:.4f}")
    print(f"    Result: {match}")
    return match == "MATCH"


def compare_detection(py_data, java_data, label):
    """对比文本检测结果"""
    py_count = py_data.get("detection", {}).get("num_regions", 0)
    java_count = java_data.get("detection_count", 0)

    match = "MATCH" if py_count == java_count else "DIFF"

    print(f"\n  [{label}] Text Detection:")
    print(f"    Python: {py_count} regions")
    print(f"    Java:   {java_count} regions")
    print(f"    Result: {match}")

    # Compare polygons if counts match
    if py_count == java_count and py_count > 0:
        py_polys = py_data.get("detection", {}).get("polygons", [])
        java_recs = java_data.get("recognition", [])
        poly_match_count = 0
        for i in range(min(len(py_polys), len(java_recs))):
            java_poly = java_recs[i].get("polygon", [])
            py_poly = py_polys[i]
            # Compare as flat lists (order may differ)
            if java_poly and py_poly:
                # Flatten and compare approximate values
                java_flat = [c for row in java_poly for c in row]
                py_flat = [c for row in py_poly for c in row]
                if len(java_flat) == len(py_flat):
                    all_close = all(abs(j - p) <= 2 for j, p in zip(java_flat, py_flat))
                    if all_close:
                        poly_match_count += 1
        print(f"    Polygon match (tolerance=2px): {poly_match_count}/{py_count}")

    return match == "MATCH"


def compare_textline_orientation(py_data, java_data, label):
    """对比文本行方向分类结果"""
    py_angles = py_data.get("textline_orientation", {}).get("angles", [])
    java_recs = java_data.get("recognition", [])

    print(f"\n  [{label}] Text Line Orientation:")
    match_count = 0
    diff_count = 0
    for i in range(min(len(py_angles), len(java_recs))):
        py_angle = py_angles[i]
        py_label = "180_degree" if py_angle == 1 else "0_degree"
        java_ori = java_recs[i].get("textline_orientation", "unknown")
        java_score = java_recs[i].get("textline_orientation_score", 0)

        match = "MATCH" if py_label == java_ori else "DIFF"
        if py_label == java_ori:
            match_count += 1
        else:
            diff_count += 1
        print(f"    Line {i}: Python={py_label}, Java={java_ori}(score={java_score:.4f}) [{match}]")

    print(f"    Summary: {match_count} match, {diff_count} diff")
    return diff_count == 0


def compare_recognition(py_data, java_data, label):
    """对比文本识别结果"""
    py_recs = py_data.get("recognition", [])
    java_recs = java_data.get("recognition", [])

    print(f"\n  [{label}] Text Recognition:")
    match_count = 0
    diff_count = 0
    for i in range(max(len(py_recs), len(java_recs))):
        py_text = py_recs[i]["text"] if i < len(py_recs) else "<missing>"
        py_conf = py_recs[i]["confidence"] if i < len(py_recs) else 0
        java_text = java_recs[i]["text"] if i < len(java_recs) else "<missing>"
        java_conf = java_recs[i]["confidence"] if i < len(java_recs) else 0

        match = "MATCH" if py_text == java_text else "DIFF"
        if py_text == java_text:
            match_count += 1
        else:
            diff_count += 1

        conf_diff = abs(py_conf - java_conf)
        print(f"    Line {i}: Python=\"{py_text}\"(conf={py_conf:.4f}), Java=\"{java_text}\"(conf={java_conf:.4f}), conf_diff={conf_diff:.4f} [{match}]")

    print(f"    Summary: {match_count} match, {diff_count} diff")
    return diff_count == 0


def compare_cpu_gpu(java_cpu, java_gpu):
    """对比Java CPU和GPU结果"""
    print(f"\n{'='*60}")
    print("Java CPU vs GPU Comparison:")
    print(f"{'='*60}")

    cpu_recs = java_cpu.get("recognition", [])
    gpu_recs = java_gpu.get("recognition", [])

    match_count = 0
    diff_count = 0
    for i in range(max(len(cpu_recs), len(gpu_recs))):
        cpu_text = cpu_recs[i]["text"] if i < len(cpu_recs) else "<missing>"
        gpu_text = gpu_recs[i]["text"] if i < len(gpu_recs) else "<missing>"
        cpu_conf = cpu_recs[i]["confidence"] if i < len(cpu_recs) else 0
        gpu_conf = gpu_recs[i]["confidence"] if i < len(gpu_recs) else 0

        match = "MATCH" if cpu_text == gpu_text else "DIFF"
        if cpu_text == gpu_text:
            match_count += 1
        else:
            diff_count += 1

        conf_diff = abs(cpu_conf - gpu_conf)
        print(f"  Line {i}: CPU=\"{cpu_text}\"(conf={cpu_conf:.4f}), GPU=\"{gpu_text}\"(conf={gpu_conf:.4f}), conf_diff={conf_diff:.4f} [{match}]")

    print(f"\n  Summary: {match_count} match, {diff_count} diff")
    return diff_count == 0


def main():
    # Load results
    print("Loading results...")
    py_data = load_json(PYTHON_RESULT)
    java_cpu = load_json(JAVA_CPU_RESULT)
    java_gpu = load_json(JAVA_GPU_RESULT)

    py_cpu = py_data.get("cpu", {})
    py_gpu = py_data.get("gpu", {})

    # ── Python CPU vs Java CPU ──
    print(f"\n{'='*60}")
    print("Python (PaddleX) CPU vs Java CPU:")
    print(f"{'='*60}")

    all_match = True
    all_match &= compare_doc_orientation(py_cpu, java_cpu, "CPU")
    all_match &= compare_detection(py_cpu, java_cpu, "CPU")
    all_match &= compare_textline_orientation(py_cpu, java_cpu, "CPU")
    all_match &= compare_recognition(py_cpu, java_cpu, "CPU")

    # ── Python GPU vs Java GPU ──
    if py_gpu:
        print(f"\n{'='*60}")
        print("Python (PaddleX) GPU vs Java GPU:")
        print(f"{'='*60}")

        all_match &= compare_doc_orientation(py_gpu, java_gpu, "GPU")
        all_match &= compare_detection(py_gpu, java_gpu, "GPU")
        all_match &= compare_textline_orientation(py_gpu, java_gpu, "GPU")
        all_match &= compare_recognition(py_gpu, java_gpu, "GPU")

    # ── Java CPU vs GPU ──
    compare_cpu_gpu(java_cpu, java_gpu)

    # ── Python CPU vs GPU ──
    if py_gpu:
        print(f"\n{'='*60}")
        print("Python (PaddleX) CPU vs GPU:")
        print(f"{'='*60}")

        py_cpu_recs = py_cpu.get("recognition", [])
        py_gpu_recs = py_gpu.get("recognition", [])
        match_count = 0
        for i in range(min(len(py_cpu_recs), len(py_gpu_recs))):
            match = "MATCH" if py_cpu_recs[i]["text"] == py_gpu_recs[i]["text"] else "DIFF"
            if py_cpu_recs[i]["text"] == py_gpu_recs[i]["text"]:
                match_count += 1
            print(f"  Line {i}: CPU=\"{py_cpu_recs[i]['text']}\", GPU=\"{py_gpu_recs[i]['text']}\" [{match}]")
        print(f"  Summary: {match_count}/{len(py_cpu_recs)} match")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print("FINAL SUMMARY:")
    print(f"{'='*60}")
    if all_match:
        print("  ALL COMPARISONS PASSED!")
    else:
        print("  SOME DIFFERENCES FOUND - see details above")
        print("  Note: Small differences in textline orientation and recognition")
        print("  are expected due to floating-point precision differences between")
        print("  Python (PaddleX) and Java (ONNX Runtime) implementations.")


if __name__ == "__main__":
    main()
