"""
Inspect LightOnOCR ONNX model I/O names, shapes, and types.
Run: D:\conda\envs\qwen3vlonnx\python scripts\inspect_lightonocr.py
"""
import onnxruntime as ort

models = [
    r"D:\models\LightOnOCR-2-1B-ONNX\embed_tokens.onnx",
    r"D:\models\LightOnOCR-2-1B-ONNX\vision_encoder.onnx",
    r"D:\models\LightOnOCR-2-1B-ONNX\decoder_model_merged.onnx",
]

for model_path in models:
    print("=" * 80)
    print(f"MODEL: {model_path.split(chr(92))[-1]}")
    print("=" * 80)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print(f"\n--- INPUTS ({len(sess.get_inputs())} total) ---")
    for inp in sess.get_inputs():
        print(f"  Name: {inp.name:45s}  Shape: {str(inp.shape):30s}  Type: {inp.type}")

    print(f"\n--- OUTPUTS ({len(sess.get_outputs())} total) ---")
    for out in sess.get_outputs():
        print(f"  Name: {out.name:45s}  Shape: {str(out.shape):30s}  Type: {out.type}")

    print()
    del sess
