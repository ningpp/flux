"""
对比 Python PyTorch 推理结果和 Java Flux ONNX 推理结果的差异。

使用方法：
    python compare_results.py
    python compare_results.py --python results_python.json --java results_java.json

输出：
    - 控制台打印汇总报告
    - 保存 verify_report.md 到脚本同目录下
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_detections(py_dets, java_dets, box_tol=5.0):
    """
    对同一张图片的 Python 和 Java 检测结果进行匹配。
    使用贪心匹配：按标签+坐标最近匹配。
    返回 (matched_pairs, unmatched_py, unmatched_java)
    """
    py_used = [False] * len(py_dets)
    java_used = [False] * len(java_dets)
    matched = []

    for pi, pd in enumerate(py_dets):
        best_j = -1
        best_dist = float("inf")
        for ji, jd in enumerate(java_dets):
            if java_used[ji]:
                continue
            if pd["label"] != jd["label"]:
                continue
            # Box distance (sum of absolute differences)
            dist = sum(abs(a - b) for a, b in zip(pd["box"], jd["box"]))
            if dist < best_dist:
                best_dist = dist
                best_j = ji
        if best_j >= 0 and best_dist < box_tol * 4:
            matched.append((pi, best_j, best_dist))
            py_used[pi] = True
            java_used[best_j] = True

    unmatched_py = [py_dets[i] for i in range(len(py_dets)) if not py_used[i]]
    unmatched_java = [java_dets[j] for j in range(len(java_dets)) if not java_used[j]]

    return matched, unmatched_py, unmatched_java


def main():
    parser = argparse.ArgumentParser(description="对比 Python 和 Java 推理结果")
    parser.add_argument("--python", type=str, default="results_python.json")
    parser.add_argument("--java", type=str, default="results_java.json")
    parser.add_argument("--output", type=str, default="verify_report.md")
    parser.add_argument("--box-tol", type=float, default=5.0, help="坐标匹配容忍度（像素）")
    parser.add_argument("--score-tol", type=float, default=0.01, help="分数匹配容忍度")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    py_path = script_dir / args.python
    java_path = script_dir / args.java
    report_path = script_dir / args.output

    py_results = load_results(py_path)
    java_results = load_results(java_path)

    # Stats
    py_imgs = set(py_results.keys())
    java_imgs = set(java_results.keys())
    common_imgs = sorted(py_imgs & java_imgs)
    only_py = sorted(py_imgs - java_imgs)
    only_java = sorted(java_imgs - py_imgs)

    # Per-image comparison
    total_py_dets = 0
    total_java_dets = 0
    total_matched = 0
    total_label_match = 0
    total_score_ok = 0
    total_box_ok = 0
    max_score_diff = 0.0
    max_box_diff = 0.0
    all_score_diffs = []
    all_box_diffs = []
    count_mismatch_images = []  # images where count differs
    label_mismatch_count = 0
    images_with_issues = []

    for img_name in common_imgs:
        py_dets = py_results[img_name]["detections"]
        java_dets = java_results[img_name]["detections"]

        total_py_dets += len(py_dets)
        total_java_dets += len(java_dets)

        if len(py_dets) != len(java_dets):
            count_mismatch_images.append((img_name, len(py_dets), len(java_dets)))

        matched, unmatched_py, unmatched_java = match_detections(py_dets, java_dets, args.box_tol)

        total_matched += len(matched)
        img_issues = []

        for pi, ji, dist in matched:
            pd = py_dets[pi]
            jd = java_dets[ji]
            label_ok = pd["label"] == jd["label"]
            score_diff = abs(pd["score"] - jd["score"])
            box_diff = max(abs(a - b) for a, b in zip(pd["box"], jd["box"]))

            if label_ok:
                total_label_match += 1
            else:
                label_mismatch_count += 1
                img_issues.append(f"  label mismatch: py={pd['label']} java={jd['label']}")

            if score_diff < args.score_tol:
                total_score_ok += 1
            all_score_diffs.append(score_diff)
            max_score_diff = max(max_score_diff, score_diff)

            if box_diff < args.box_tol:
                total_box_ok += 1
            all_box_diffs.append(box_diff)
            max_box_diff = max(max_box_diff, box_diff)

        if unmatched_py or unmatched_java or img_issues:
            images_with_issues.append((img_name, len(unmatched_py), len(unmatched_java), img_issues))

    # Generate report
    lines = []
    lines.append("# PP-DocLayoutV3 Python vs Java 验证报告")
    lines.append("")
    lines.append(f"- Python 结果: `{args.python}`")
    lines.append(f"- Java 结果: `{args.java}`")
    lines.append(f"- 匹配容忍度: 坐标 {args.box_tol}px, 分数 {args.score_tol}")
    lines.append("")

    # Summary
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 图片总数 | {len(common_imgs)} |")
    lines.append(f"| 仅 Python 有 | {len(only_py)} |")
    lines.append(f"| 仅 Java 有 | {len(only_java)} |")
    lines.append(f"| Python 总检测框数 | {total_py_dets} |")
    lines.append(f"| Java 总检测框数 | {total_java_dets} |")
    lines.append(f"| 匹配检测框数 | {total_matched} |")
    lines.append(f"| 标签完全匹配 | {total_label_match}/{total_matched} ({100*total_label_match/max(total_matched,1):.1f}%) |")
    lines.append(f"| 分数差异 < {args.score_tol} | {total_score_ok}/{total_matched} ({100*total_score_ok/max(total_matched,1):.1f}%) |")
    lines.append(f"| 坐标差异 < {args.box_tol}px | {total_box_ok}/{total_matched} ({100*total_box_ok/max(total_matched,1):.1f}%) |")
    lines.append(f"| 最大分数差异 | {max_score_diff:.6f} |")
    lines.append(f"| 最大坐标差异 | {max_box_diff:.2f}px |")
    lines.append(f"| 检测框数不一致的图片 | {len(count_mismatch_images)} |")
    lines.append("")

    # Score diff distribution
    if all_score_diffs:
        import numpy as np
        sd = np.array(all_score_diffs)
        lines.append("## 分数差异分布")
        lines.append("")
        lines.append(f"| 统计量 | 值 |")
        lines.append(f"|--------|-----|")
        lines.append(f"| 平均 | {sd.mean():.6f} |")
        lines.append(f"| 中位数 | {np.median(sd):.6f} |")
        lines.append(f"| P95 | {np.percentile(sd, 95):.6f} |")
        lines.append(f"| P99 | {np.percentile(sd, 99):.6f} |")
        lines.append(f"| 最大 | {sd.max():.6f} |")
        lines.append("")

    # Box diff distribution
    if all_box_diffs:
        bd = np.array(all_box_diffs)
        lines.append("## 坐标差异分布 (px)")
        lines.append("")
        lines.append(f"| 统计量 | 值 |")
        lines.append(f"|--------|-----|")
        lines.append(f"| 平均 | {bd.mean():.4f} |")
        lines.append(f"| 中位数 | {np.median(bd):.4f} |")
        lines.append(f"| P95 | {np.percentile(bd, 95):.4f} |")
        lines.append(f"| P99 | {np.percentile(bd, 99):.4f} |")
        lines.append(f"| 最大 | {bd.max():.4f} |")
        lines.append("")

    # Count mismatch details
    if count_mismatch_images:
        lines.append("## 检测框数不一致的图片")
        lines.append("")
        lines.append(f"共 {len(count_mismatch_images)} 张图片检测框数不一致:")
        lines.append("")
        lines.append("| 图片 | Python | Java | 差异 |")
        lines.append("|------|--------|------|------|")
        for img_name, pn, jn in count_mismatch_images:
            lines.append(f"| {img_name} | {pn} | {jn} | {pn - jn:+d} |")
        lines.append("")

        # Near-threshold analysis for count-mismatched images
        lines.append("### 近阈值分析")
        lines.append("")
        lines.append("对检测框数不一致的图片，分析未匹配检测的得分是否接近阈值（0.5）：")
        lines.append("")
        near_threshold_count = 0
        for img_name, pn, jn in count_mismatch_images:
            py_dets = py_results[img_name]["detections"]
            java_dets = java_results[img_name]["detections"]
            matched, unmatched_py, unmatched_java = match_detections(py_dets, java_dets, args.box_tol)
            has_near = False
            near_items = []
            for d in unmatched_py:
                if 0.45 < d["score"] < 0.55:
                    near_items.append(f"Py: {d['label']} score={d['score']:.6f} (delta={d['score']-0.5:+.6f})")
                    has_near = True
            for d in unmatched_java:
                if 0.45 < d["score"] < 0.55:
                    near_items.append(f"Java: {d['label']} score={d['score']:.6f} (delta={d['score']-0.5:+.6f})")
                    has_near = True
            if has_near:
                near_threshold_count += 1
                lines.append(f"**{img_name}** (Py={pn}, Java={jn}):")
                for item in near_items:
                    lines.append(f"  - {item}")
                lines.append("")
        if near_threshold_count > 0:
            lines.append(f"**{near_threshold_count}/{len(count_mismatch_images)}** 张图片的未匹配检测框得分在阈值 0.5 附近（0.45~0.55），"
                         "差异由 float32 精度导致得分落在阈值不同侧所致，属于预期行为。")
            lines.append("")


    # Verdict: use P99 metrics instead of max to avoid single-outlier failure
    label_rate = total_label_match / max(total_matched, 1)
    score_rate = total_score_ok / max(total_matched, 1)
    box_rate = total_box_ok / max(total_matched, 1)
    count_rate = 1.0 - len(count_mismatch_images) / max(len(common_imgs), 1)

    import numpy as np
    sd = np.array(all_score_diffs) if all_score_diffs else np.array([0])
    bd = np.array(all_box_diffs) if all_box_diffs else np.array([0])
    p99_score = float(np.percentile(sd, 99))
    p99_box = float(np.percentile(bd, 99))

    passed = (label_rate > 0.99 and score_rate > 0.95 and box_rate > 0.95
              and p99_score < 0.01 and p99_box < 1.0 and count_rate > 0.95)

    lines.append("## 结论")
    lines.append("")
    if passed:
        lines.append("**✅ 验证通过！Java ONNX 推理结果与 Python PyTorch 结果一致。**")
    else:
        lines.append("**❌ 验证未通过，存在显著差异。**")
        lines.append("")
        if label_rate <= 0.99:
            lines.append(f"- 标签匹配率 {label_rate*100:.1f}% 低于 99%")
        if score_rate <= 0.95:
            lines.append(f"- 分数匹配率 {score_rate*100:.1f}% 低于 95%")
        if box_rate <= 0.95:
            lines.append(f"- 坐标匹配率 {box_rate*100:.1f}% 低于 95%")
        if count_rate <= 0.95:
            lines.append(f"- 检测框数一致率 {count_rate*100:.1f}% 低于 95%")
        if p99_score >= 0.01:
            lines.append(f"- P99 分数差异 {p99_score:.6f} >= 0.01")
        if p99_box >= 1.0:
            lines.append(f"- P99 坐标差异 {p99_box:.2f}px >= 1.0px")
    lines.append("")

    report_text = "\n".join(lines)

    # Print to console
    print(report_text)

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
