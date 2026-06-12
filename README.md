# Flux

⚠️ WARNING: This project is in its early stages of development.

Do not use in production.

[![License](https://img.shields.io/badge/License-Apache%202-blue.svg)](LICENSE)


## **Special Thanks**
1. [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX)
2. [Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)
3. [breezedeus/Pix2Text](https://github.com/breezedeus/Pix2Text)
4. [NormXU/nougat-latex-ocr](https://github.com/NormXU/nougat-latex-ocr)
5. [huggingface/transformers](https://github.com/huggingface/transformers)
6. [bytedance/Dolphin](https://github.com/bytedance/Dolphin/blob/v1.5)
7. [docling-project/docling](https://github.com/docling-project/docling)
8. [huggingface/optimum](https://github.com/huggingface/optimum)
9. [OleehyO/TexTeller](https://github.com/OleehyO/TexTeller)


### Support Model Category
| Model Category      | Status |
|:--------------------|:------:|
| Layout              |   ✅    |
| Text Detection      |   ✅    |
| Text Recognition    |   ✅    |
| Formula Recognition |   ✅    |
| Table Recognition   |   ✅    |
| Doc Orientation     |   ✅    |
| Text Line Orientation |   ✅    |
| Multimodal OCR      |   ✅    |



### Support Model List

#### Layout Model
| Model                                                                      | CPU | CUDA |
|:---------------------------------------------------------------------------|:---:|:----:|
| [docling-layout-egret-large](https://huggingface.co/ningpp/docling-layout-egret-large-ONNX)  |  ✅  |  ✅   |
| [docling-layout-egret-medium](https://huggingface.co/ningpp/docling-layout-egret-medium-ONNX) |  ✅  |  ✅   |
| [docling-layout-egret-xlarge](https://huggingface.co/ningpp/docling-layout-egret-xlarge-ONNX) |  ✅  |  ✅   |
| [docling-layout-heron](https://huggingface.co/ningpp/docling-layout-heron-ONNX)        |  ✅  |  ✅   |
| [docling-layout-heron-101](https://huggingface.co/ningpp/docling-layout-heron-101-ONNX)    |  ✅  |  ✅   |
| PP-DocLayoutV2              |  ✅  |  ✅   |
| [PP-DocLayoutV3](https://huggingface.co/ningpp/PP-DocLayoutV3-ONNX)              |  ✅  |  ✅   |
| PP-DocLayout_plus-L         |  ✅  |  ✅   |
| PP-DocLayout-L              |  ✅  |  ✅   |
| PP-DocLayout-M              |  ✅  |  ✅   |
| PP-DocLayout-S              |  ✅  |  ✅   |
| PicoDet-S_layout_17cls      |  ✅  |  ✅   |
| PicoDet-L_layout_17cls      |  ✅  |  ✅   |
| RT-DETR-H_layout_17cls      |  ✅  |  ✅   |


#### Text Detection Model
| Model                           | CPU | CUDA |
|:--------------------------------|:---:|:----:|
| [PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) |  ✅  |  ✅   |
| [PP-OCRv6_small_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx)   |  ✅  |  ✅   |
| [PP-OCRv6_tiny_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)     |  ✅  |  ✅   |
| PP-OCRv5_server_det             |  ✅  |  ✅   |
| PP-OCRv5_mobile_det |  ✅  |  ✅   |
| PP-OCRv4_server_det |  ✅  |  ✅   |
| PP-OCRv4_mobile_det |  ✅  |  ✅   |


#### Text Recognition Model
| Model                           | CPU | CUDA |
|:--------------------------------|:---:|:----:|
| [PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) |  ✅  |  ✅   |
| [PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx)   |  ✅  |  ✅   |
| [PP-OCRv6_tiny_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)     |  ✅  |  ✅   |
| PP-OCRv5_server_rec             |  ✅  |  ✅   |
| PP-OCRv5_mobile_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec_doc |  ✅  |  ✅   |
| PP-OCRv4_mobile_rec     |  ✅  |  ✅   |


#### Formula Recognition Model
| Model                | CPU | CUDA |
|:---------------------|:---:|:----:|
| [CodeFormulaV2](https://huggingface.co/ningpp/CodeFormulaV2-ONNX)        |  ✅  |  ✅   |
| Dolphin              |  ✅  |  ✅   |
| Dolphin-1.5          |  ✅  |  ✅   |
| [Falcon-OCR](https://huggingface.co/ningpp/Falcon-OCR-ONNX)           |  ✅  |  ✅   |
| [GOT-OCR-2.0](https://huggingface.co/ningpp/GOT-OCR-2.0-ONNX)          |  ✅  |  ✅   |
| granite-docling-258M |  ✅  |  ✅   |
| nougat-latex-base    |  ✅  |  ✅   |
| pix2text-mfr         |  ✅  |  ✅   |
| pix2text-mfr-1.5     |  ✅  |  ✅   |
| PP-FormulaNet-S      |  ✅  |  ❌   |
| [PP-FormulaNet-L](https://huggingface.co/ningpp/PP-FormulaNet-L-ONNX)      |  ✅  |  ✅   |
| PP-FormulaNet_plus-S |  ✅  |  ❌   |
| PP-FormulaNet_plus-M |  ✅  |  ❌   |
| [PP-FormulaNet_plus-L](https://huggingface.co/ningpp/PP-FormulaNet_plus-L-ONNX) |  ✅  |  ✅   |
| TexTeller            |  ✅  |  ✅   |
| unirec-0.1b          |  ✅  |  ✅   |


#### Table Recognition Model
| Model       | CPU | CUDA |
|:------------|:---:|:----:|
| Dolphin     |  ✅  |  ✅   |
| Dolphin-1.5 |  ✅  |  ✅   |
| Falcon-OCR  |  ✅  |  ✅   |
| unirec-0.1b |  ✅  |  ✅   |


#### Doc Orientation Model
| Model                 | CPU | CUDA |
|:----------------------|:---:|:----:|
| [PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx) |  ✅  |  ✅   |


#### Text Line Orientation Model
| Model                 | CPU | CUDA |
|:----------------------|:---:|:----:|
| [PP-LCNet_x1_0_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx) |  ✅  |  ✅   |
| [PP-LCNet_x0_25_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx) |  ✅  |  ✅   |


#### Multimodal OCR Model
| Model                            | CPU | CUDA |
|:---------------------------------|:---:|:----:|
| [GLM-OCR](https://huggingface.co/ningpp/GLM-OCR-ONNX)                          |  ✅  |  ✅   |
| LightOnOCR-2-1B                  |  ✅  |  ✅   |
| LightOnOCR-2-1B-ONNX             |  ✅  |  ✅   |
| llava-onevision-qwen2-0.5b-ov-hf |  ✅  |  ✅   |


### OCR Pipeline

The OCR Pipeline combines multiple models to perform end-to-end text extraction from document images. It orchestrates the following steps:

1. **Document Orientation Classification** (optional) — Detects and corrects the overall document rotation (0°/90°/180°/270°).
2. **Text Detection** — Locates text regions in the image and returns bounding polygons.
3. **Text Line Orientation Classification** (optional) — Detects whether each text line is upside down (0° or 180°) and rotates it if needed.
4. **Text Recognition** — Recognizes text content from each detected text line.

#### Usage

```java
try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
    // Required models
    TextDetectionModel detModel = new TextDetectionModel(modelDir, "PP-OCRv6_medium_det", env, gpuIndex);
    TextRecognitionModel recModel = new TextRecognitionModel(modelDir, "PP-OCRv6_medium_rec", env, gpuIndex);

    // Optional models (pass null to skip)
    DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(modelDir, "PP-LCNet_x1_0_doc_ori", env, gpuIndex);
    TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(modelDir, "PP-LCNet_x1_0_textline_ori", env, gpuIndex);

    OCRPipeline pipeline = new OCRPipeline(detModel, recModel, docOriModel, textLineOriModel);

    Map<String, Object> params = new HashMap<>();
    params.put("recognitionBatchSize", 1);

    List<OCRPipelineResult> results = pipeline.predictFile("image.png", params);

    for (OCRPipelineResult result : results) {
        String text = result.recResults().stream()
            .map(r -> r.text())
            .collect(Collectors.joining());
        System.out.println(text);
    }
}
```

#### Parameters

| Parameter | Type | Default | Description |
|:----------|:----:|:-------:|:------------|
| `recognitionBatchSize` | Integer | 1 | Batch size for text recognition inference |

#### OCRPipelineResult Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `detPolys` | `int[][]` | Detected text region polygon coordinates |
| `recResults` | `List<RecognitionResult>` | Recognized text and confidence scores |
| `docOrientationLabel` | `String` | Document orientation label (e.g., "0", "90", "180", "270") |
| `docOrientationScore` | `float` | Document orientation classification confidence |
| `textLineOrientationLabel` | `String` | Text line orientation label (e.g., "0_degree", "180_degree") |
| `textLineOrientationScore` | `float` | Text line orientation classification confidence |
