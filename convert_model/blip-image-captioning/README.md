# Convert BLIP models to ONNX

## blip-image-captioning-base

- Torch model dir: `D:\models\blip-image-captioning-base`
- Converted onnx dir: `D:\models\onnx\blip-image-captioning-base`

Scripts:
- Convert: `python convert_blip_to_onnx.py`
- Validate: `python validate_onnx.py`

## blip-image-captioning-large

- Torch model dir: `D:\models\blip-image-captioning-large`
- Converted onnx dir: `D:\models\onnx\blip-image-captioning-large`

Scripts:
- Convert: `python convert_blip_large_to_onnx.py`
- Validate: `python validate_blip_large.py`

## Inference

Run `inference_onnx.py` to generate captions using both Base and Large ONNX models.

```bash
python inference_onnx.py
```

## Environment

Use conda env `qwen3vlonnx` (`D:\conda\envs\qwen3vlonnx`)

## Validation Images

- `D:\tmp\img-2026-02-07-120018.png`
- `D:\tmp\img-2026-02-07-120114.png`
