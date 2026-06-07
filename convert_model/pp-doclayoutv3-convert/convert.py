"""
PP-DocLayoutV3 模型转换为 ONNX 格式，并使用 imgs 文件夹批量推理验证正确性。

使用方法（在 conda paddlex 环境下）：
    python convert.py              # 转换 + 验证
    python convert.py --export     # 仅导出 ONNX
    python convert.py --verify     # 仅验证
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ────────────────────── 路径配置 ──────────────────────
TORCH_DIR = r"D:\models\PP-DocLayoutV3_safetensors"
ONNX_DIR = r"D:\models\layout\PP-DocLayoutV3"
IMGS_DIR = Path(__file__).parent / "imgs"
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")

# ────────────────────── 标签映射 ──────────────────────
# inference.yml 中的 label_list（25个类别）
LABEL_LIST = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]


def export_onnx():
    """将 PP-DocLayoutV3 PyTorch 模型导出为 ONNX（含 out_masks）"""
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"[1/3] 加载模型: {TORCH_DIR}")
    model = AutoModelForObjectDetection.from_pretrained(TORCH_DIR)
    model.eval()

    print(f"[2/3] 加载预处理器: {TORCH_DIR}")
    processor = AutoImageProcessor.from_pretrained(TORCH_DIR)

    # 用虚拟图片确定输入 shape
    dummy_img = Image.new("RGB", (800, 800))
    dummy_inputs = processor(images=[dummy_img], return_tensors="pt")
    pixel_values = dummy_inputs["pixel_values"]  # (1, 3, 800, 800)

    # 构建动态轴：支持 batch 和空间维度动态
    dynamic_axes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "logits":       {0: "batch"},
        "pred_boxes":   {0: "batch"},
        "order_logits": {0: "batch"},
        "out_masks":    {0: "batch"},
    }

    print(f"[3/3] 导出 ONNX → {ONNX_PATH}")
    torch.onnx.export(
        model,
        (pixel_values,),
        ONNX_PATH,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "order_logits", "out_masks"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX 导出完成: {ONNX_PATH}")

    # 裁剪 ONNX 只保留需要的4个输出
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    keep_names = {"logits", "pred_boxes", "order_logits", "out_masks"}
    new_outputs = [o for o in onnx_model.graph.output if o.name in keep_names]
    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(new_outputs)

    onnx.save(onnx_model, ONNX_PATH)
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX 输出: {[o.name for o in onnx_model.graph.output]}")
    print("ONNX 模型校验通过")

    fsize = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"ONNX 文件大小: {fsize:.1f} MB")


def verify():
    """使用 ONNX Runtime 对 imgs 目录下图片批量推理，并与 PyTorch 结果对比验证"""
    import onnxruntime as ort
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import (
        PPDocLayoutV3ForObjectDetectionOutput,
    )

    # 收集测试图片
    img_paths = sorted(IMGS_DIR.glob("*.png")) + sorted(IMGS_DIR.glob("*.jpg"))
    if not img_paths:
        print(f"错误: 在 {IMGS_DIR} 下未找到任何图片")
        sys.exit(1)

    print(f"加载 PyTorch 模型...")
    model = AutoModelForObjectDetection.from_pretrained(TORCH_DIR)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(TORCH_DIR)

    print(f"加载 ONNX 模型: {ONNX_PATH}")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print(f"ONNX 输入: {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"ONNX 输出: {[(o.name, o.shape) for o in session.get_outputs()]}")

    images = [Image.open(p).convert("RGB") for p in img_paths]
    original_sizes = [img.size[::-1] for img in images]  # (H, W)

    # ───── PyTorch 推理 ─────
    pt_inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        pt_outputs = model(**pt_inputs)

    pt_results = processor.post_process_object_detection(
        pt_outputs,
        threshold=0.5,
        target_sizes=torch.tensor(original_sizes),
    )

    # ───── ONNX 推理 ─────
    ort_inputs = {"pixel_values": pt_inputs["pixel_values"].numpy()}
    ort_outputs = session.run(None, ort_inputs)
    output_names = [o.name for o in session.get_outputs()]
    ort_map = dict(zip(output_names, ort_outputs))

    # 将 ONNX 输出封装为 transformers 兼容对象
    onnx_outputs = PPDocLayoutV3ForObjectDetectionOutput(
        logits=torch.from_numpy(ort_map["logits"]),
        pred_boxes=torch.from_numpy(ort_map["pred_boxes"]),
        order_logits=torch.from_numpy(ort_map["order_logits"]),
        out_masks=torch.from_numpy(ort_map["out_masks"]),
    )
    onnx_results = processor.post_process_object_detection(
        onnx_outputs,
        threshold=0.5,
        target_sizes=torch.tensor(original_sizes),
    )

    # ───── 对比结果 ─────
    print(f"\n{'='*60}")
    print(f"验证图片: {len(images)} 张 (batch 推理)")
    print(f"{'='*60}")

    all_pass = True
    for i, (img_path, pt_res, onnx_res) in enumerate(zip(img_paths, pt_results, onnx_results)):
        pt_boxes  = pt_res["boxes"].numpy()
        pt_scores = pt_res["scores"].numpy()
        pt_labels = pt_res["labels"].numpy()

        onnx_boxes  = onnx_res["boxes"].numpy()
        onnx_scores = onnx_res["scores"].numpy()
        onnx_labels = onnx_res["labels"].numpy()

        n_pt   = len(pt_scores)
        n_onnx = len(onnx_scores)

        count_match = n_pt == n_onnx
        labels_match = np.array_equal(pt_labels, onnx_labels) if count_match else False
        score_diff = np.max(np.abs(pt_scores - onnx_scores)) if count_match and n_pt > 0 else 0.0
        box_diff = np.max(np.abs(pt_boxes - onnx_boxes)) if count_match and n_pt > 0 else 0.0

        # 对比 polygon_points
        polygon_match = True
        if count_match and n_pt > 0 and "polygon_points" in pt_res and "polygon_points" in onnx_res:
            pt_polys = pt_res["polygon_points"]
            onnx_polys = onnx_res["polygon_points"]
            if len(pt_polys) == len(onnx_polys):
                for pp, op in zip(pt_polys, onnx_polys):
                    if pp is not None and op is not None:
                        if not np.allclose(pp, op, atol=1.0):
                            polygon_match = False
                            break

        passed = count_match and labels_match and score_diff < 0.01 and box_diff < 1.0 and polygon_match
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] 图片 {i+1}: {img_path.name}")
        print(f"  PyTorch 检测框: {n_pt}  |  ONNX 检测框: {n_onnx}")
        if n_pt > 0:
            print(f"  标签一致: {labels_match}")
            print(f"  最大分数差异: {score_diff:.6f}")
            print(f"  最大坐标差异: {box_diff:.4f}")
            print(f"  多边形一致: {polygon_match}")
            for j in range(min(5, n_onnx)):
                label_name = LABEL_LIST[onnx_labels[j]] if onnx_labels[j] < len(LABEL_LIST) else str(onnx_labels[j])
                box = onnx_boxes[j]
                print(f"    [{j+1}] {label_name}: score={onnx_scores[j]:.3f}  box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]")
            if n_onnx > 5:
                print(f"    ... 共 {n_onnx} 个检测框")

    print(f"\n{'='*60}")
    if all_pass:
        print("全部验证通过！ONNX 模型与 PyTorch 模型推理结果一致。")
    else:
        print("验证未通过，请检查 ONNX 导出是否正确。")
    print(f"{'='*60}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="PP-DocLayoutV3 -> ONNX 转换与验证")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--export", action="store_true", help="仅导出 ONNX")
    group.add_argument("--verify", action="store_true", help="仅验证 ONNX")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.export:
        export_onnx()
    else:
        export_onnx()
        verify()


if __name__ == "__main__":
    main()
