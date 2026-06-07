"""
PP-DocLayoutV3 Python (PyTorch/transformers) 批量推理脚本。
对指定目录下所有图片进行推理，输出每张图片的检测结果到 JSON 文件。

使用方法（在 conda paddlex 环境下）：
    conda run -n paddlex python verify_python.py
    conda run -n paddlex python verify_python.py --img-dir D:\data\DocLayNet-v1.2-imgs
    conda run -n paddlex python verify_python.py --batch-size 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

TORCH_DIR = r"D:\models\PP-DocLayoutV3_safetensors"
LABEL_LIST = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]


def main():
    parser = argparse.ArgumentParser(description="PP-DocLayoutV3 Python 批量推理")
    parser.add_argument("--img-dir", type=str, default=r"D:\data\DocLayNet-v1.2-imgs",
                        help="图片目录")
    parser.add_argument("--output", type=str, default="results_python.json",
                        help="输出 JSON 文件名（保存在脚本同目录下）")
    parser.add_argument("--batch-size", type=int, default=4, help="推理 batch 大小")
    parser.add_argument("--threshold", type=float, default=0.5, help="检测阈值")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    img_dir = Path(args.img_dir)

    # Collect images
    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        print(f"错误: 在 {img_dir} 下未找到任何图片")
        sys.exit(1)
    print(f"共 {len(img_paths)} 张图片")

    # Load model & processor
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    print("加载 PyTorch 模型...")
    model = AutoModelForObjectDetection.from_pretrained(TORCH_DIR)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(TORCH_DIR)

    # Batch inference
    all_results = {}
    total_images = len(img_paths)
    t0 = time.time()

    for start in tqdm(range(0, total_images, args.batch_size), desc="Python 推理"):
        batch_paths = img_paths[start:start + args.batch_size]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"  跳过无法读取的图片 {p.name}: {e}")

        if not images:
            continue

        original_sizes = [img.size[::-1] for img in images]  # (H, W)
        pt_inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            pt_outputs = model(**pt_inputs)

        pt_results = processor.post_process_object_detection(
            pt_outputs,
            threshold=args.threshold,
            target_sizes=torch.tensor(original_sizes),
        )

        for img_path, pt_res, orig_size in zip(valid_paths, pt_results, original_sizes):
            detections = []
            boxes = pt_res["boxes"].numpy()
            scores = pt_res["scores"].numpy()
            labels = pt_res["labels"].numpy()

            for j in range(len(scores)):
                box = boxes[j].tolist()
                det = {
                    "label": LABEL_LIST[labels[j]] if labels[j] < len(LABEL_LIST) else str(labels[j]),
                    "score": round(float(scores[j]), 6),
                    "box": [round(float(v), 2) for v in box],
                }
                detections.append(det)

            all_results[img_path.name] = {
                "width": orig_size[1],
                "height": orig_size[0],
                "detections": detections,
            }

    elapsed = time.time() - t0
    print(f"\n推理完成: {total_images} 张图片, 耗时 {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
