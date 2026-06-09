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
| Model               | CPU | CUDA |
|:--------------------|:---:|:----:|
| PP-OCRv5_server_det |  ✅  |  ✅   |
| PP-OCRv5_mobile_det |  ✅  |  ✅   |
| PP-OCRv4_server_det |  ✅  |  ✅   |
| PP-OCRv4_mobile_det |  ✅  |  ✅   |


#### Text Recognition Model
| Model                   | CPU | CUDA |
|:------------------------|:---:|:----:|
| PP-OCRv5_server_rec     |  ✅  |  ✅   |
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
| [PP-LCNet_x1_0_doc_ori](https://huggingface.co/ningpp/PP-LCNet_x1_0_doc_ori-ONNX) |  ✅  |  ✅   |


#### Multimodal OCR Model
| Model                            | CPU | CUDA |
|:---------------------------------|:---:|:----:|
| [GLM-OCR](https://huggingface.co/ningpp/GLM-OCR-ONNX)                          |  ✅  |  ✅   |
| LightOnOCR-2-1B                  |  ✅  |  ✅   |
| LightOnOCR-2-1B-ONNX             |  ✅  |  ✅   |
| llava-onevision-qwen2-0.5b-ov-hf |  ✅  |  ✅   |
