"""
系统性排查 Python vs Java textline orientation 推理差异
逐步对比：图片读取、Resize、Normalize、ToCHW、ONNX输入Tensor
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
import struct

IMG_DIR = r"E:\textline-ori-imgs"
OUTPUT_DIR = r"D:\code\flux\scripts\pp-ocrv6\output_textline_ori\debug"
MODEL_DIR = r"D:\models\PP-LCNet_x1_0_textline_ori_onnx"

# 预处理参数
RESIZE_W, RESIZE_H = 160, 80
SCALE = 1.0 / 255.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def save_tensor_binary(tensor, filepath):
    """保存numpy tensor为二进制文件（用于Java端对比）"""
    tensor.astype(np.float32).tofile(filepath)


def save_tensor_json(tensor, filepath):
    """保存numpy tensor为JSON（前20个值+统计信息）"""
    import json
    flat = tensor.flatten()
    data = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "first_20": flat[:20].tolist(),
        "last_20": flat[-20:].tolist(),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "sum": float(flat.sum()),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def debug_preprocess(img_path, img_name):
    """逐步预处理并保存每步中间结果"""
    print(f"\n{'='*70}")
    print(f"图片: {img_name}")
    print(f"{'='*70}")

    # Step 1: 读取
    img_bgr = cv2.imread(img_path)
    print(f"  Step1 读取: shape={img_bgr.shape}, dtype={img_bgr.dtype}")
    print(f"    BGR [0,0]={img_bgr[0,0]}, [0,1]={img_bgr[0,1]}")
    print(f"    BGR [40,80]={img_bgr[min(40,img_bgr.shape[0]-1),min(80,img_bgr.shape[1]-1)]}")

    # Step 2: BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  Step2 RGB: [0,0]={img_rgb[0,0]}, [0,1]={img_rgb[0,1]}")

    # Step 3: Resize (width=160, height=80)
    img_resized = cv2.resize(img_rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    print(f"  Step3 Resize: shape={img_resized.shape}")
    print(f"    [0,0]={img_resized[0,0]}, [0,1]={img_resized[0,1]}")
    print(f"    [40,80]={img_resized[40,80]}")
    print(f"    [79,159]={img_resized[79,159]}")

    # Step 4: float32 + /255
    img_float = img_resized.astype(np.float32) * SCALE
    print(f"  Step4 /255: [0,0]={img_float[0,0]}, [0,1]={img_float[0,1]}")
    print(f"    [40,80]={img_float[40,80]}")

    # Step 5: Normalize (x - mean) / std
    img_norm = (img_float - MEAN) / STD
    print(f"  Step5 Normalize: [0,0]={img_norm[0,0]}, [0,1]={img_norm[0,1]}")
    print(f"    [40,80]={img_norm[40,80]}")
    print(f"    min={img_norm.min():.6f}, max={img_norm.max():.6f}, mean={img_norm.mean():.6f}")

    # Step 6: HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)
    print(f"  Step6 CHW: shape={img_chw.shape}")
    print(f"    C0[0,0:5]={img_chw[0,0,:5]}")
    print(f"    C1[0,0:5]={img_chw[1,0,:5]}")
    print(f"    C2[0,0:5]={img_chw[2,0,:5]}")
    print(f"    C0[40,80]={img_chw[0,40,80]}")
    print(f"    C1[40,80]={img_chw[1,40,80]}")
    print(f"    C2[40,80]={img_chw[2,40,80]}")

    # Step 7: Add batch dimension
    img_batch = img_chw[np.newaxis, :]
    print(f"  Step7 Batch: shape={img_batch.shape}")

    # 保存中间结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = img_name.replace(".png", "")

    # 保存最终输入tensor（二进制，供Java对比）
    save_tensor_binary(img_batch, os.path.join(OUTPUT_DIR, f"{base}_input_tensor.bin"))
    save_tensor_json(img_batch, os.path.join(OUTPUT_DIR, f"{base}_input_tensor.json"))

    # 保存resize后的uint8图像（供Java对比）
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_resized_rgb.png"), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    # 保存normalize后的HWC float数据
    save_tensor_binary(img_norm, os.path.join(OUTPUT_DIR, f"{base}_norm_hwc.bin"))

    # 保存CHW float数据
    save_tensor_binary(img_chw, os.path.join(OUTPUT_DIR, f"{base}_norm_chw.bin"))

    return img_batch


def debug_inference(img_batch, img_name):
    """使用CPU推理，保存输出"""
    model_path = os.path.join(MODEL_DIR, "inference.onnx")

    # 强制使用CPU
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # CPU推理
    outputs = session.run(None, {input_name: img_batch})
    logits = outputs[0]

    base = img_name.replace(".png", "")
    print(f"  推理结果(CPU): {logits}")
    print(f"    sum={logits.sum():.6f} (应为~1.0如果已是概率)")

    # 保存输出
    import json
    with open(os.path.join(OUTPUT_DIR, f"{base}_cpu_output.json"), "w") as f:
        json.dump({"logits": logits.tolist(), "provider": "CPU"}, f, indent=2)

    return logits


def main():
    img_files = sorted([
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    for img_path in img_files:
        img_name = os.path.basename(img_path)
        img_batch = debug_preprocess(img_path, img_name)
        debug_inference(img_batch, img_name)

    print(f"\n\n所有调试数据已保存到: {OUTPUT_DIR}")
    print("请在Java端实现相同的调试输出进行对比")


if __name__ == "__main__":
    main()
