"""
Inspect Qwen3-VL ONNX models — list I/O names, shapes, dtypes.
Optionally consolidate external data into a single file.

Run: D:\conda\envs\qwen3vlonnx\python scripts\inspect_qwen3vl.py
"""
import os
import onnx
import onnxruntime as ort

ONNX_DIR = r"D:\models\onnx\Qwen3-VL-2B-Instruct"


def inspect_model(name: str):
    path = os.path.join(ONNX_DIR, name)
    print(f"\n{'=' * 60}")
    print(f"Model: {name}  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
    print(f"{'=' * 60}")

    # Use ORT to get I/O info (handles external data automatically)
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

        print("\nInputs:")
        for inp in sess.get_inputs():
            print(f"  {inp.name:45s}  shape={inp.shape}  dtype={inp.type}")

        print("\nOutputs:")
        for out in sess.get_outputs():
            print(f"  {out.name:45s}  shape={out.shape}  dtype={out.type}")

        del sess
    except Exception as e:
        print(f"  ERROR loading with ORT: {e}")


def consolidate_external_data(name: str):
    """Consolidate external data files into a single .onnx.data file."""
    path = os.path.join(ONNX_DIR, name)
    data_file = name + ".data"
    out_path = os.path.join(ONNX_DIR, name + ".tmp")

    print(f"\nConsolidating external data for {name}...")
    model = onnx.load(path, load_external_data=True)
    onnx.save_model(
        model,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
    )

    # Replace original
    os.replace(out_path, path)
    # Remove old external data files (individual weight files)
    for f in os.listdir(ONNX_DIR):
        fpath = os.path.join(ONNX_DIR, f)
        if os.path.isfile(fpath) and f not in (
            name, data_file, "vision_encoder.onnx", "embed_tokens.onnx",
            "config.json", "generation_config.json", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json",
        ) and not f.endswith(".onnx.data"):
            try:
                os.remove(fpath)
            except Exception:
                pass

    data_path = os.path.join(ONNX_DIR, data_file)
    if os.path.exists(data_path):
        print(f"  Consolidated: {data_file} ({os.path.getsize(data_path) / 1024 / 1024:.1f} MB)")


def main():
    # Inspect all models
    for name in ["vision_encoder.onnx", "embed_tokens.onnx", "decoder_model_merged.onnx"]:
        inspect_model(name)

    # Consolidate decoder external data
    consolidate_external_data("decoder_model_merged.onnx")

    print("\n\nFinal files:")
    for f in sorted(os.listdir(ONNX_DIR)):
        fpath = os.path.join(ONNX_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024
            print(f"  {f:50s}  {size:8.1f} MB")


if __name__ == "__main__":
    main()
