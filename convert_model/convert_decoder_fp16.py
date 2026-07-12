import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

model = onnx.load(r"D:\models\onnx\GOT-OCR-2.0\decoder_model.onnx")
fp16 = convert_float_to_float16(model, keep_io_types=False, min_positive_val=5.96e-08, max_finite_val=65504.0)
onnx.save(fp16, r"D:\models\onnx\GOT-OCR-2.0-FP16\decoder_model.onnx")
print("saved decoder fp16")
