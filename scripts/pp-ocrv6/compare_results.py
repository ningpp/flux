"""
PP-OCRv6 Python vs Java 推理结果对比脚本
支持 medium/small/tiny 三个模型变体的对比
"""

import json
import math
from pathlib import Path

PYTHON_OUTPUT_DIR = Path(r"D:\code\paddle-ocr-model-v6\output")
JAVA_OUTPUT_DIR = Path(r"D:\code\paddle-ocr-model-v6\output_java")

VARIANTS = ["PP-OCRv6_medium", "PP-OCRv6_small", "PP-OCRv6_tiny"]
IMAGES = ["img-2026-01-24-193955", "img-2026-06-11-195511"]

IOU_THRESHOLD = 0.5
COORD_TOLERANCE = 5
SCORE_TOLERANCE = 0.01


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def match_regions(py_polys, java_polys):
    matches = []
    used_java = set()
    for pi, py_poly in enumerate(py_polys):
        py_bbox = poly_to_bbox(py_poly)
        best_jv, best_iou = -1, 0.0
        for ji, java_poly in enumerate(java_polys):
            if ji in used_java:
                continue
            iou = compute_iou(py_bbox, poly_to_bbox(java_poly))
            if iou > best_iou:
                best_iou, best_jv = iou, ji
        if best_iou > IOU_THRESHOLD:
            matches.append((pi, best_jv))
            used_java.add(best_jv)
    return matches


def compare_one(py_data, java_data):
    py_polys = py_data.get("dt_polys", [])
    java_polys = java_data.get("dt_polys", [])
    py_texts = py_data.get("rec_texts", [])
    py_scores = py_data.get("rec_scores", [])
    java_texts = java_data.get("rec_texts", [])
    java_scores = java_data.get("rec_scores", [])

    det_match = len(py_polys) == len(java_polys)
    matches = match_regions(py_polys, java_polys)

    text_match = 0
    max_coord = 0
    max_score_diff = 0
    baseline_diffs = []
    cascade_diffs = []

    for pi, ji in matches:
        # coord diff
        region_max = 0
        for k in range(min(len(py_polys[pi]), len(java_polys[ji]))):
            dx = abs(py_polys[pi][k][0] - java_polys[ji][k][0])
            dy = abs(py_polys[pi][k][1] - java_polys[ji][k][1])
            region_max = max(region_max, dx, dy)
        max_coord = max(max_coord, region_max)

        # text
        pt = py_texts[pi] if pi < len(py_texts) else ""
        jt = java_texts[ji] if ji < len(java_texts) else ""
        if pt == jt:
            text_match += 1

        # score
        ps = py_scores[pi] if pi < len(py_scores) else 0.0
        js = java_scores[ji] if ji < len(java_scores) else 0.0
        sd = abs(ps - js)
        max_score_diff = max(max_score_diff, sd)

        if region_max == 0:
            baseline_diffs.append(sd)
        else:
            cascade_diffs.append((region_max, sd))

    avg_baseline = sum(baseline_diffs) / len(baseline_diffs) if baseline_diffs else 0

    return {
        "det_count_match": det_match,
        "py_det_count": len(py_polys),
        "java_det_count": len(java_polys),
        "match_count": len(matches),
        "text_match": text_match,
        "max_coord": max_coord,
        "max_score_diff": max_score_diff,
        "avg_baseline": avg_baseline,
        "matches": matches,
        "py_texts": py_texts,
        "java_texts": java_texts,
        "py_scores": py_scores,
        "java_scores": java_scores,
        "py_polys": py_polys,
        "java_polys": java_polys,
        "cascade_diffs": cascade_diffs,
    }


def main():
    all_ok = True
    summary_rows = []

    for variant in VARIANTS:
        print(f"\n{'='*80}")
        print(f"模型: {variant}")
        print(f"{'='*80}")

        for img_name in IMAGES:
            py_file = PYTHON_OUTPUT_DIR / f"{variant}_{img_name}_res.json"
            java_file = JAVA_OUTPUT_DIR / f"{variant}_{img_name}_res.json"

            if not py_file.exists() or not java_file.exists():
                print(f"  ⚠️ 缺少文件: py={py_file.exists()}, java={java_file.exists()}")
                continue

            py_data = load_json(py_file)
            java_data = load_json(java_file)
            r = compare_one(py_data, java_data)

            det_icon = "✅" if r["det_count_match"] else "❌"
            coord_icon = "✅" if r["max_coord"] <= COORD_TOLERANCE else "❌"
            score_icon = "✅" if r["max_score_diff"] <= SCORE_TOLERANCE else "❌"
            text_icon = "✅" if r["text_match"] == len(r["matches"]) else "❌"

            print(f"\n  图片: {img_name}")
            print(f"  检测框: Python={r['py_det_count']} Java={r['java_det_count']} {det_icon}")
            print(f"  区域匹配: {r['match_count']}/{r['py_det_count']}")
            print(f"  文本匹配: {r['text_match']}/{r['match_count']} {text_icon}")
            print(f"  最大坐标差: {r['max_coord']}px {coord_icon}")
            print(f"  最大分数差: {r['max_score_diff']:.6f} {score_icon}")
            print(f"  ONNX基线: ~{r['avg_baseline']:.4f}")

            # 逐行对比
            print(f"\n  {'#':>3} {'文本':^6} {'坐标':^6} {'分数差异':>10} {'Python':>10} {'Java':>10} {'来源'}")
            print(f"  {'-'*70}")
            for pi, ji in r["matches"]:
                pt = r["py_texts"][pi] if pi < len(r["py_texts"]) else ""
                jt = r["java_texts"][ji] if ji < len(r["java_texts"]) else ""
                tm = "✅" if pt == jt else "❌"

                region_max = 0
                for k in range(min(len(r["py_polys"][pi]), len(r["java_polys"][ji]))):
                    dx = abs(r["py_polys"][pi][k][0] - r["java_polys"][ji][k][0])
                    dy = abs(r["py_polys"][pi][k][1] - r["java_polys"][ji][k][1])
                    region_max = max(region_max, dx, dy)
                cm = "✅" if region_max <= COORD_TOLERANCE else "❌"

                ps = r["py_scores"][pi] if pi < len(r["py_scores"]) else 0
                js = r["java_scores"][ji] if ji < len(r["java_scores"]) else 0
                sd = abs(ps - js)
                sm = "✅" if sd <= SCORE_TOLERANCE else "❌"

                src = "A(ORT)" if region_max == 0 else "A+B"
                print(f"  {pi:>3} {tm:^6} {region_max:>2}px{cm:^3} {sd:>10.6f}{sm:^2} {ps:>10.6f} {js:>10.6f} {src}")

            ok = (r["det_count_match"] and r["text_match"] == len(r["matches"])
                  and r["max_coord"] <= COORD_TOLERANCE and r["max_score_diff"] <= SCORE_TOLERANCE)
            if not ok:
                all_ok = False

            summary_rows.append((variant, img_name, r["max_coord"], r["max_score_diff"], r["avg_baseline"], ok))

    # Summary
    print(f"\n{'='*80}")
    print(f"总结")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'图片':<30} {'最大坐标差':>10} {'最大分数差':>12} {'ORT基线':>10} {'结果'}")
    print("-" * 90)
    for variant, img, mc, msd, bl, ok in summary_rows:
        icon = "✅" if ok else "❌"
        print(f"{variant:<20} {img:<30} {mc:>8}px {msd:>12.6f} {bl:>10.4f} {icon}")

    print()
    if all_ok:
        print("✅ 所有模型变体的Python和Java推理结果在容忍范围内一致!")
    else:
        print("❌ 存在超出容忍范围的差异")


if __name__ == "__main__":
    main()
