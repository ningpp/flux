"""
PP-LCNet textline orientation Python vs Java 推理结果对比脚本
比较两个实现的分类标签和置信度分数差异

差异来源说明：
- CPU vs CPU: 差异极小 (<0.01%)，由浮点精度导致
- CUDA vs CPU: 差异可达0.5%，由ONNX Runtime CUDA/CPU计算路径不同导致
- 建议对比时两端使用相同的推理设备(CPU)
"""

import json
import os
from pathlib import Path

PYTHON_OUTPUT_DIR = Path(r"D:\code\flux\scripts\pp-ocrv6\output_textline_ori")
JAVA_OUTPUT_DIR = Path(r"D:\code\flux\scripts\pp-ocrv6\output_textline_ori")

MODELS = ["PP-LCNet_x1_0_textline_ori", "PP-LCNet_x0_25_textline_ori"]

# 分数差异阈值：0.5%
SCORE_DIFF_THRESHOLD = 0.005


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results():
    all_ok = True
    summary_rows = []

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"模型: {model_name}")
        print(f"{'='*80}")

        py_file = PYTHON_OUTPUT_DIR / f"{model_name}_results.json"
        java_file = JAVA_OUTPUT_DIR / f"{model_name}_java_results.json"

        if not py_file.exists():
            print(f"  ❌ Python结果文件不存在: {py_file}")
            all_ok = False
            continue
        if not java_file.exists():
            print(f"  ❌ Java结果文件不存在: {java_file}")
            all_ok = False
            continue

        py_data = load_json(py_file)
        java_data = load_json(java_file)

        for img_name, py_result in py_data.items():
            if img_name not in java_data:
                print(f"  ⚠️ Java结果中缺少图片: {img_name}")
                continue

            java_result = java_data[img_name]
            py_label = py_result["results"][0]["label"]
            py_score = py_result["results"][0]["score"]
            java_label = java_result["label"]
            java_score = java_result["score"]

            label_match = py_label == java_label
            score_diff = abs(py_score - java_score)
            label_icon = "✅" if label_match else "❌"
            score_icon = "✅" if score_diff <= SCORE_DIFF_THRESHOLD else "❌"

            # 检查是否大于0.5%
            score_diff_pct = score_diff * 100
            exceeds_threshold = score_diff > SCORE_DIFF_THRESHOLD

            print(f"  {img_name}")
            print(f"    Python: label={py_label}, score={py_score:.6f}")
            print(f"    Java:   label={java_label}, score={java_score:.6f}")
            print(f"    标签匹配: {label_icon}  分数差异: {score_diff:.6f} ({score_diff_pct:.3f}%) {score_icon}")

            if not label_match or exceeds_threshold:
                all_ok = False

            summary_rows.append((model_name, img_name, label_match, score_diff, score_diff_pct, not exceeds_threshold and label_match))

    # Summary
    print(f"\n{'='*80}")
    print(f"总结")
    print(f"{'='*80}")
    print(f"{'模型':<30} {'图片':<45} {'标签':^6} {'分数差异':>10} {'差异%':>8} {'结果'}")
    print("-" * 110)
    for model, img, lm, sd, sdp, ok in summary_rows:
        icon = "✅" if ok else "❌"
        lm_str = "✅" if lm else "❌"
        print(f"{model:<30} {img:<45} {lm_str:^6} {sd:>10.6f} {sdp:>7.3f}% {icon}")

    print()
    if all_ok:
        print("✅ 所有模型的Python和Java推理结果差异在0.5%以内!")
    else:
        print("❌ 存在超出0.5%差异的结果，需要分析原因")
        analyze_differences(summary_rows)


def analyze_differences(summary_rows):
    """分析差异原因"""
    print(f"\n{'='*80}")
    print("差异原因分析")
    print(f"{'='*80}")

    has_diff = False
    for model, img, lm, sd, sdp, ok in summary_rows:
        if not ok:
            has_diff = True
            print(f"\n  模型: {model}, 图片: {img}")
            print(f"  分数差异: {sd:.6f} ({sdp:.3f}%)")
            if not lm:
                print(f"  ❌ 标签不一致! 这表明预处理或推理存在显著差异")
            else:
                print(f"  标签一致但分数差异超阈值")
            print(f"  可能原因:")
            print(f"    1. 图像Resize插值算法差异 (Python OpenCV vs Java OpenCV)")
            print(f"    2. 浮点精度差异 (Python float64 vs Java float32)")
            print(f"    3. Normalize计算顺序差异")
            print(f"    4. ONNX Runtime版本差异 (Python vs Java)")

    if not has_diff:
        print("  无显著差异需要分析")


if __name__ == "__main__":
    compare_results()
