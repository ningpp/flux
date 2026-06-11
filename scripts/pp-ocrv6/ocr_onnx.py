"""
PP-OCRv6 ONNX模型推理脚本
使用本地的ONNX模型进行文字检测和文字识别
"""

import os
from paddleocr import PaddleOCR

# 模型路径
DET_MODEL_DIR = r"D:\models\PP-OCRv6_medium_det_onnx"
REC_MODEL_DIR = r"D:\models\PP-OCRv6_medium_rec_onnx"

# 图片目录
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    # 初始化PaddleOCR，使用ONNX推理引擎，指定本地ONNX模型路径
    ocr = PaddleOCR(
        text_detection_model_dir=DET_MODEL_DIR,
        text_recognition_model_dir=REC_MODEL_DIR,
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0",
    )

    # 获取imgs目录下所有图片
    img_files = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    if not img_files:
        print(f"在 {IMG_DIR} 中未找到图片文件")
        return

    print(f"找到 {len(img_files)} 张图片，开始OCR识别...\n")

    # 逐张图片进行推理
    for img_path in img_files:
        print(f"{'='*60}")
        print(f"图片: {os.path.basename(img_path)}")
        print(f"{'='*60}")

        result = ocr.predict(img_path)

        for res in result:
            res.print()

            # 保存可视化结果
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            res.save_to_img(OUTPUT_DIR)
            res.save_to_json(OUTPUT_DIR)

        print()

    print(f"所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
