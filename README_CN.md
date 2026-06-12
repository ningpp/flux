# Flux

⚠️ 警告：本项目处于早期开发阶段，请勿用于生产环境。

[![License](https://img.shields.io/badge/License-Apache%202-blue.svg)](LICENSE)


## **特别感谢**
1. [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX)
2. [Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)
3. [breezedeus/Pix2Text](https://github.com/breezedeus/Pix2Text)
4. [NormXU/nougat-latex-ocr](https://github.com/NormXU/nougat-latex-ocr)
5. [huggingface/transformers](https://github.com/huggingface/transformers)
6. [bytedance/Dolphin](https://github.com/bytedance/Dolphin/blob/v1.5)
7. [docling-project/docling](https://github.com/docling-project/docling)
8. [huggingface/optimum](https://github.com/huggingface/optimum)
9. [OleehyO/TexTeller](https://github.com/OleehyO/TexTeller)


### 支持的模型类别
| 模型类别       | 状态 |
|:-------------|:----:|
| 版面分析       |  ✅   |
| 文本检测       |  ✅   |
| 文本识别       |  ✅   |
| 公式识别       |  ✅   |
| 表格识别       |  ✅   |
| 文档方向分类    |  ✅   |
| 文本行方向分类  |  ✅   |
| 多模态OCR     |  ✅   |



### 支持的模型列表

#### 版面分析模型
| 模型                          | CPU | CUDA |
|:------------------------------|:---:|:----:|
| [docling-layout-egret-large](https://huggingface.co/ningpp/docling-layout-egret-large-ONNX)    |  ✅  |  ✅   |
| [docling-layout-egret-medium](https://huggingface.co/ningpp/docling-layout-egret-medium-ONNX)   |  ✅  |  ✅   |
| [docling-layout-egret-xlarge](https://huggingface.co/ningpp/docling-layout-egret-xlarge-ONNX)   |  ✅  |  ✅   |
| [docling-layout-heron](https://huggingface.co/ningpp/docling-layout-heron-ONNX)          |  ✅  |  ✅   |
| [docling-layout-heron-101](https://huggingface.co/ningpp/docling-layout-heron-101-ONNX)      |  ✅  |  ✅   |
| PP-DocLayoutV2                |  ✅  |  ✅   |
| [PP-DocLayoutV3](https://huggingface.co/ningpp/PP-DocLayoutV3-ONNX)                |  ✅  |  ✅   |
| PP-DocLayout_plus-L           |  ✅  |  ✅   |
| PP-DocLayout-L                |  ✅  |  ✅   |
| PP-DocLayout-M                |  ✅  |  ✅   |
| PP-DocLayout-S                |  ✅  |  ✅   |
| PicoDet-S_layout_17cls        |  ✅  |  ✅   |
| PicoDet-L_layout_17cls        |  ✅  |  ✅   |
| RT-DETR-H_layout_17cls        |  ✅  |  ✅   |


#### 文本检测模型
| 模型                             | CPU | CUDA |
|:--------------------------------|:---:|:----:|
| [PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) |  ✅  |  ✅   |
| [PP-OCRv6_small_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx)   |  ✅  |  ✅   |
| [PP-OCRv6_tiny_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)     |  ✅  |  ✅   |
| PP-OCRv5_server_det              |  ✅  |  ✅   |
| PP-OCRv5_mobile_det |  ✅  |  ✅   |
| PP-OCRv4_server_det |  ✅  |  ✅   |
| PP-OCRv4_mobile_det |  ✅  |  ✅   |


#### 文本识别模型
| 模型                             | CPU | CUDA |
|:--------------------------------|:---:|:----:|
| [PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) |  ✅  |  ✅   |
| [PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx)   |  ✅  |  ✅   |
| [PP-OCRv6_tiny_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)     |  ✅  |  ✅   |
| PP-OCRv5_server_rec              |  ✅  |  ✅   |
| PP-OCRv5_mobile_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec_doc |  ✅  |  ✅   |
| PP-OCRv4_mobile_rec     |  ✅  |  ✅   |


#### 公式识别模型
| 模型                  | CPU | CUDA |
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


#### 表格识别模型
| 模型        | CPU | CUDA |
|:-----------|:---:|:----:|
| Dolphin     |  ✅  |  ✅   |
| Dolphin-1.5 |  ✅  |  ✅   |
| Falcon-OCR  |  ✅  |  ✅   |
| unirec-0.1b |  ✅  |  ✅   |


#### 文档方向分类模型
| 模型                  | CPU | CUDA |
|:---------------------|:---:|:----:|
| [PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx) |  ✅  |  ✅   |


#### 文本行方向分类模型
| 模型                  | CPU | CUDA |
|:---------------------|:---:|:----:|
| [PP-LCNet_x1_0_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx) |  ✅  |  ✅   |
| [PP-LCNet_x0_25_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx) |  ✅  |  ✅   |


#### 多模态OCR模型
| 模型                              | CPU | CUDA |
|:---------------------------------|:---:|:----:|
| [GLM-OCR](https://huggingface.co/ningpp/GLM-OCR-ONNX)                          |  ✅  |  ✅   |
| LightOnOCR-2-1B                  |  ✅  |  ✅   |
| LightOnOCR-2-1B-ONNX             |  ✅  |  ✅   |
| llava-onevision-qwen2-0.5b-ov-hf |  ✅  |  ✅   |


### OCR Pipeline

OCR Pipeline 将多个模型组合，实现从文档图片到文本的端到端提取。流程如下：

1. **文档方向分类**（可选）— 检测并纠正文档整体旋转方向（0°/90°/180°/270°）。
2. **文本检测** — 定位图片中的文本区域，返回多边形边界框。
3. **文本行方向分类**（可选）— 检测每个文本行是否倒置（0° 或 180°），若倒置则旋转纠正。
4. **文本识别** — 对每个检测到的文本行进行文字识别。

#### 使用示例

```java
try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
    // 必需模型
    TextDetectionModel detModel = new TextDetectionModel(modelDir, "PP-OCRv6_medium_det", env, gpuIndex);
    TextRecognitionModel recModel = new TextRecognitionModel(modelDir, "PP-OCRv6_medium_rec", env, gpuIndex);

    // 可选模型（传 null 可跳过）
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

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:----:|:-----:|:-----|
| `recognitionBatchSize` | Integer | 1 | 文本识别推理的批大小 |

#### OCRPipelineResult 字段

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `detPolys` | `int[][]` | 检测到的文本区域多边形坐标 |
| `recResults` | `List<RecognitionResult>` | 识别的文本及置信度 |
| `docOrientationLabel` | `String` | 文档方向标签（如 "0"、"90"、"180"、"270"） |
| `docOrientationScore` | `float` | 文档方向分类置信度 |
| `textLineOrientationLabel` | `String` | 文本行方向标签（如 "0_degree"、"180_degree"） |
| `textLineOrientationScore` | `float` | 文本行方向分类置信度 |
