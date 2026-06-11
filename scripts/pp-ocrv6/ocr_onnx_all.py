"""
PP-OCRv6 small/tiny ONNX模型推理脚本
为所有v6模型变体生成Python基准结果
"""

import os
from paddleocr import PaddleOCR

MODEL_VARIANTS = [
    ("PP-OCRv6_medium", r"D:\models\PP-OCRv6_medium_det_onnx", r"D:\models\PP-OCRv6_medium_rec_onnx"),
    ("PP-OCRv6_small", r"D:\models\PP-OCRv6_small_det_onnx", r"D:\models\PP-OCRv6_small_rec_onnx"),
    ("PP-OCRv6_tiny", r"D:\models\PP-OCRv6_tiny_det_onnx", r"D:\models\PP-OCRv6_tiny_rec_onnx"),
]

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    img_files = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    if not img_files:
        print(f"在 {IMG_DIR} 中未找到图片文件")
        return

    for variant_name, det_model_dir, rec_model_dir in MODEL_VARIANTS:
        print(f"\n{'='*60}")
        print(f"模型: {variant_name}")
        print(f"{'='*60}")

        ocr = PaddleOCR(
            text_detection_model_dir=det_model_dir,
            text_recognition_model_dir=rec_model_dir,
            text_detection_model_name=variant_name + "_det",
            text_recognition_model_name=variant_name + "_rec",
            engine="onnxruntime",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0",
        )

        for img_path in img_files:
            print(f"  处理: {os.path.basename(img_path)}")
            result = ocr.predict(img_path)
            for res in result:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                # 用variant_name前缀区分不同模型的输出
                base_name = os.path.basename(img_path).rsplit(".", 1)[0]
                res.save_to_json(OUTPUT_DIR)
                # 重命名以包含variant信息
                json_path = os.path.join(OUTPUT_DIR, f"{base_name}_res.json")
                new_path = os.path.join(OUTPUT_DIR, f"{variant_name}_{base_name}_res.json")
                if os.path.exists(new_path):
                    os.remove(new_path)
                if os.path.exists(json_path):
                    os.rename(json_path, new_path)
                    print(f"  保存: {new_path}")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
