"""
PP-LCNet textline orientation ONNX推理脚本
支持 PP-LCNet_x1_0_textline_ori 和 PP-LCNet_x0_25_textline_ori 模型
使用纯ONNX Runtime进行推理，预处理/后处理手动实现，确保与Java实现一致
"""

import os
import json
import argparse
import numpy as np
import cv2
import onnxruntime as ort

MODEL_CONFIGS = {
    "PP-LCNet_x1_0_textline_ori": {
        "model_dir": r"D:\models\PP-LCNet_x1_0_textline_ori_onnx",
        "resize": (160, 80),  # (width, height)
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scale": 1.0 / 255.0,
        "labels": ["0_degree", "180_degree"],
    },
    "PP-LCNet_x0_25_textline_ori": {
        "model_dir": r"D:\models\PP-LCNet_x0_25_textline_ori_onnx",
        "resize": (160, 80),  # (width, height)
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scale": 1.0 / 255.0,
        "labels": ["0_degree", "180_degree"],
    },
}


def preprocess(img_path, resize_wh, scale, mean, std):
    """预处理：Read -> BGR2RGB -> Resize -> Normalize -> ToCHW"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to (width, height)
    img = cv2.resize(img, resize_wh, interpolation=cv2.INTER_LINEAR)

    # Normalize: (pixel * scale - mean) / std
    img = img.astype(np.float32) * scale
    img = (img - np.array(mean)) / np.array(std)

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # Add batch dimension
    img = img[np.newaxis, :].astype(np.float32)
    return img


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def postprocess(logits, labels):
    """后处理：模型输出已经是softmax概率，直接取top1"""
    probs = logits
    top1_idx = np.argmax(probs, axis=-1)
    results = []
    for i, idx in enumerate(top1_idx):
        results.append({
            "label": labels[idx],
            "score": float(probs[i][idx]),
            "all_probs": {labels[j]: float(probs[i][j]) for j in range(len(labels))}
        })
    return results


def run_inference(model_name, img_dir, output_dir):
    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(config["model_dir"], "inference.onnx")

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    # 创建ONNX会话 - 使用GPU确保与Java一致
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"模型: {model_name}")
    print(f"输入: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"输出: {[o.name for o in session.get_outputs()]}")
    print(f"Provider: {session.get_providers()}")

    # 获取测试图片
    img_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    if not img_files:
        print(f"在 {img_dir} 中未找到图片文件")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for img_path in img_files:
        img_name = os.path.basename(img_path)
        print(f"\n  处理: {img_name}")

        # 预处理
        input_data = preprocess(
            img_path,
            config["resize"],
            config["scale"],
            config["mean"],
            config["std"]
        )

        # 推理
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]

        # 后处理
        results = postprocess(logits, config["labels"])

        for r in results:
            print(f"    标签: {r['label']}, 置信度: {r['score']:.6f}")
            print(f"    所有概率: {r['all_probs']}")

        all_results[img_name] = {
            "model_name": model_name,
            "image": img_name,
            "results": results,
            "raw_logits": logits.tolist()
        }

    # 保存结果
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="TextLine Orientation ONNX推理")
    parser.add_argument("--model", type=str, default="all",
                        help="模型名称: PP-LCNet_x1_0_textline_ori, PP-LCNet_x0_25_textline_ori, 或 all")
    args = parser.parse_args()

    img_dir = r"E:\textline-ori-imgs"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_textline_ori")

    if args.model == "all":
        for model_name in MODEL_CONFIGS:
            run_inference(model_name, img_dir, output_dir)
    else:
        run_inference(args.model, img_dir, output_dir)


if __name__ == "__main__":
    main()
