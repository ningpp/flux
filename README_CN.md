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
| 多模态OCR     |  ✅   |



### 支持的模型列表

#### 版面分析模型
| 模型                          | CPU | CUDA |
|:------------------------------|:---:|:----:|
| docling-layout-egret-large    |  ✅  |  ✅   |
| docling-layout-egret-medium   |  ✅  |  ✅   |
| docling-layout-egret-xlarge   |  ✅  |  ✅   |
| docling-layout-heron          |  ✅  |  ✅   |
| docling-layout-heron-101      |  ✅  |  ✅   |
| PP-DocLayoutV2                |  ✅  |  ✅   |
| PP-DocLayoutV3                |  ✅  |  ✅   |
| PP-DocLayout_plus-L           |  ✅  |  ✅   |
| PP-DocLayout-L                |  ✅  |  ✅   |
| PP-DocLayout-M                |  ✅  |  ✅   |
| PP-DocLayout-S                |  ✅  |  ✅   |
| PicoDet-S_layout_17cls        |  ✅  |  ✅   |
| PicoDet-L_layout_17cls        |  ✅  |  ✅   |
| RT-DETR-H_layout_17cls        |  ✅  |  ✅   |


#### 文本检测模型
| 模型                | CPU | CUDA |
|:-------------------|:---:|:----:|
| PP-OCRv5_server_det |  ✅  |  ✅   |
| PP-OCRv5_mobile_det |  ✅  |  ✅   |
| PP-OCRv4_server_det |  ✅  |  ✅   |
| PP-OCRv4_mobile_det |  ✅  |  ✅   |


#### 文本识别模型
| 模型                    | CPU | CUDA |
|:-----------------------|:---:|:----:|
| PP-OCRv5_server_rec     |  ✅  |  ✅   |
| PP-OCRv5_mobile_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec     |  ✅  |  ✅   |
| PP-OCRv4_server_rec_doc |  ✅  |  ✅   |
| PP-OCRv4_mobile_rec     |  ✅  |  ✅   |


#### 公式识别模型
| 模型                  | CPU | CUDA |
|:---------------------|:---:|:----:|
| CodeFormulaV2        |  ✅  |  ✅   |
| Dolphin              |  ✅  |  ✅   |
| Dolphin-1.5          |  ✅  |  ✅   |
| Falcon-OCR           |  ✅  |  ✅   |
| GOT-OCR-2.0          |  ✅  |  ✅   |
| granite-docling-258M |  ✅  |  ✅   |
| nougat-latex-base    |  ✅  |  ✅   |
| pix2text-mfr         |  ✅  |  ✅   |
| pix2text-mfr-1.5     |  ✅  |  ✅   |
| PP-FormulaNet-S      |  ✅  |  ❌   |
| PP-FormulaNet-L      |  ✅  |  ✅   |
| PP-FormulaNet_plus-S |  ✅  |  ❌   |
| PP-FormulaNet_plus-M |  ✅  |  ❌   |
| PP-FormulaNet_plus-L |  ✅  |  ✅   |
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
| PP-LCNet_x1_0_doc_ori |  ✅  |  ✅   |


#### 多模态OCR模型
| 模型                              | CPU | CUDA |
|:---------------------------------|:---:|:----:|
| GLM-OCR                          |  ✅  |  ✅   |
| LightOnOCR-2-1B                  |  ✅  |  ✅   |
| LightOnOCR-2-1B-ONNX             |  ✅  |  ✅   |
| llava-onevision-qwen2-0.5b-ov-hf |  ✅  |  ✅   |
