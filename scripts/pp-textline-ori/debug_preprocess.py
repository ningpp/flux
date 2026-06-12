"""
调试脚本：对比Python预处理后的中间张量，确认与Java一致
"""

import os
import numpy as np
import cv2
import json

IMG_DIR = r"E:\textline-ori-imgs"
OUTPUT_DIR = r"D:\code\flux\scripts\pp-ocrv6\output_textline_ori"

def preprocess_debug(img_path):
    """预处理并保存中间结果"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    print(f"  原始 shape: {img.shape}, dtype: {img.dtype}")
    print(f"  原始 [0,0] BGR: {img[0, 0]}")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  RGB [0,0]: {img[0, 0]}")

    # Resize to (160, 80) - width=160, height=80
    img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_LINEAR)
    print(f"  Resize后 shape: {img.shape}")

    # 转float32
    img_float = img.astype(np.float32) / 255.0
    print(f"  /255后 [0,0]: {img_float[0, 0]}")

    # Normalize: (pixel * scale - mean) / std = (pixel/255 - mean) / std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_float - mean) / std
    print(f"  Normalize后 [0,0]: {img_norm[0, 0]}")
    print(f"  Normalize后 [0,1]: {img_norm[0, 1]}")

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)
    print(f"  CHW shape: {img_chw.shape}")
    print(f"  CHW C=0 [0,0:5]: {img_chw[0, 0, :5]}")
    print(f"  CHW C=1 [0,0:5]: {img_chw[1, 0, :5]}")
    print(f"  CHW C=2 [0,0:5]: {img_chw[2, 0, :5]}")

    # flat data (按CHW顺序)
    flat = img_chw.flatten()
    print(f"  flat前20: {flat[:20]}")

    return {
        "shape_hwc": list(img_norm.shape),
        "pixel_0_0_rgb": img[0, 0].tolist(),
        "norm_0_0": img_norm[0, 0].tolist(),
        "norm_0_1": img_norm[0, 1].tolist(),
        "chw_c0_0_0to4": img_chw[0, 0, :5].tolist(),
        "chw_c1_0_0to4": img_chw[1, 0, :5].tolist(),
        "chw_c2_0_0to4": img_chw[2, 0, :5].tolist(),
        "flat_first20": flat[:20].tolist(),
    }


def main():
    img_files = sorted([
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    all_debug = {}
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        print(f"\n{'='*60}")
        print(f"图片: {img_name}")
        print(f"{'='*60}")
        debug_info = preprocess_debug(img_path)
        all_debug[img_name] = debug_info

    output_file = os.path.join(OUTPUT_DIR, "python_preprocess_debug.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_debug, f, ensure_ascii=False, indent=2)
    print(f"\n调试信息已保存到: {output_file}")


if __name__ == "__main__":
    main()
