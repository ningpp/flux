

# Falcon-OCR Torch → ONNX Converter

将 Falcon-OCR 的 PyTorch 权重转换为 ONNX 格式，
并验证推理结果准确度（对比每一步 Top-5 Logits）。

本仓库提供一个 ONNX 友好的 Falcon-OCR 等价实现：

- 保留原始权重、token embedding、image projector 和 22 层 Transformer。
- 将原模型中的 FlexAttention、KVCache 对象和 Triton FFN 替换为标准 ONNX 可导出的 dense attention / tensor ops。
- ONNX 输入位于模型预处理后的张量层：`tokens`、`image_patches`、`pos_t`、`pos_hw`、`attention_mask`。

## 环境要求

- Python 3.11 + conda 环境 `transformers5`
- transformers 5.8.0
- PyTorch 2.7.1
- ONNX 1.20.1
- ONNX Runtime 1.24.2
- Pillow

## 验证图片

默认使用 `imgs/` 目录中的图片验证。

## 转换

```powershell
D:\conda\envs\transformers5\python.exe convert_falcon_ocr_to_onnx.py `
  --model-dir D:\models\Falcon-OCR `
  --output-dir D:\models\Falcon-OCR-ONNX `
  --image D:\models\falcon-ocr-convert\imgs\text-2026-05-23-124542.png `
  --category text `
  --export-unified-kv `
  --skip-validation
```

完整 Top-5 验证图可按需导出并运行到 `max_length`：

```powershell
D:\conda\envs\transformers5\python.exe convert_falcon_ocr_to_onnx.py `
  --model-dir D:\models\Falcon-OCR `
  --output-dir D:\models\Falcon-OCR-ONNX `
  --image D:\models\falcon-ocr-convert\imgs\text-2026-05-23-124542.png `
  --category text `
  --validation-mode full-length `
  --full-length-generation greedy `
  --target-length 8192 `
  --progress-interval 250 `
  --report-name top5_validation_full_max8192.json
```

输出文件：

- `D:\models\Falcon-OCR-ONNX\falcon_ocr_kv.onnx`：端到端 OCR 推理使用的统一 KV-cache 模型
- `D:\models\Falcon-OCR-ONNX\falcon_ocr.onnx`：full-context Top-5 验证图
- `D:\models\Falcon-OCR-ONNX\top5_validation.json`
- tokenizer/config/processor 配套文件

## 当前验证结果

本机已完成 full-context 验证图的 ONNX Runtime CPU 验证。验证方式为 PyTorch 参考路径逐步 greedy 生成到 `max_length`，再让 `falcon_ocr.onnx` 以完整 `max_length` 序列运行一次，并逐位置对比 Top-5 logits：

- 验证样例：`imgs\text-2026-05-23-124542.png`
- prompt/category：`text`
- 初始序列长度：336
- ONNX 实际输入长度：8192
- ONNX 实际 attention mask：`1 x 8192 x 8192`
- 对比位置数：7857
- 生成步数：7856
- PyTorch greedy reference vs ONNX Runtime Top-5：全部匹配
- mismatch 数：0
- 全 logits 最大绝对误差：`0.0004811286926269531`
- 报告：`D:\models\Falcon-OCR-ONNX\top5_validation_full_max8192.json`

额外抽查也已通过：

| 样例 | 类别 | 初始序列长度 | 步数 | 最大绝对误差 |
| --- | --- | ---: | ---: | ---: |
| `formula-2026-01-18-152316.png` | `formula` | 400 | 3 | `2.8371810913085938e-05` |
| `table-2026-01-01-202211.png` | `table` | 3472 | 3 | `3.254413604736328e-05` |
| `text-2026-05-23-124711.png` | `text` | 2192 | 3 | `3.1948089599609375e-05` |

逐步 Top-5 可查看：

```powershell
Get-Content D:\models\Falcon-OCR-ONNX\top5_validation_full_max8192.json
```

## Full-Context 验证说明

`falcon_ocr.onnx` 是完整上下文 forward 图，不包含增量 KV cache，仅用于 full-length Top-5 对齐验证。full-length greedy 验证会先用 PyTorch 等价参考路径生成完整 token 序列，再让 ONNX 以该完整序列跑到 `max_length`，并对每一步的 Top-5 logits 逐项比较。

端到端部署使用 `falcon_ocr_kv_token.onnx`，该模型包含增量 KV cache，并在 ONNX 内部完成 argmax，只输出 next token id。

## KV Cache ONNX

端到端 OCR 推理推荐使用单个 token-only unified KV-cache 图：

- `falcon_ocr_kv_token.onnx`：同一个模型同时支持 prefill 和 decode，并只输出 `next_token` + `present_key_values`。
- `falcon_ocr_kv.onnx`：兼容版 unified KV 图，输出完整 next-token logits + `present_key_values`。
- prefill：输入完整图文 prompt，`past_key_values` 的序列长度为 0。
- decode：输入单个 token、当前位置、上一轮 `past_key_values`，输出 next-token id 和更新后的 `present_key_values`。
- 这样只加载一份权重，避免 `prefill.onnx` + `decode.onnx` 两个 session 重复占用内存/显存。
- token-only 图把 argmax 放进 ONNX，每步无需把 `65536` 个 float32 logits 拉回 Python。

导出命令：

```powershell
D:\conda\envs\transformers5\python.exe convert_falcon_ocr_to_onnx.py `
  --model-dir D:\models\Falcon-OCR `
  --output-dir D:\models\Falcon-OCR-ONNX `
  --image D:\models\falcon-ocr-convert\imgs\text-2026-05-23-124542.png `
  --category text `
  --export-unified-kv-token `
  --skip-validation `
  --validation-mode sample `
  --steps 1
```

## 六图端到端一致性

命令：

```powershell
D:\conda\envs\transformers5\python.exe compare_onnx_transformers_e2e.py `
  --kv-onnx D:\models\Falcon-OCR-ONNX\falcon_ocr_kv_token.onnx `
  --image-dir D:\models\falcon-ocr-convert\imgs `
  --output-dir D:\models\Falcon-OCR-ONNX `
  --max-new-tokens 512 `
  --batch-size 2 `
  --provider auto
```

说明：`transformers5` 的 PyTorch 是 CPU-only，且缺少 `triton` / `AuxRequest` 兼容项；脚本在运行时对 Transformers remote code 做 CPU 兼容补丁。ONNX Runtime 使用 `CUDAExecutionProvider, CPUExecutionProvider`。ONNX 推理循环使用真正的 batch prefill/decode 和 ORT I/O binding，使 `present_key_values` 作为 `OrtValue` 留在 ORT/CUDA 侧传递，避免每步把完整 KV cache 转回 numpy 再传入下一步。`falcon_ocr_kv_token.onnx` 只把 `next_token` 拉回 CPU，不回传完整 logits。

结果文件：

- `D:\models\Falcon-OCR-ONNX\onnx_vs_transformers_e2e.json`
- `D:\models\Falcon-OCR-ONNX\onnx_vs_transformers_perf.csv`

六张 `imgs` 图片的 ONNX batch KV-cache 推理与 Transformers 推理结果：

| 图片 | 类别 | Prompt Tokens | Batch Padded Prompt | 输出 Tokens | 文本一致 | Token 一致 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `formula-2026-01-18-152316.png` | formula | 400 | 400 | 119 | true | true |
| `formula_2025-8-2_17-28-16.jpg` | formula | 280 | 400 | 31 | true | true |
| `table-2026-01-01-202211.png` | table | 3472 | 3472 | 201 | true | true |
| `table-2026-05-23-124132.png` | table | 1168 | 3472 | 326 | true | true |
| `text-2026-05-23-124542.png` | text | 336 | 2192 | 68 | true | true |
| `text-2026-05-23-124711.png` | text | 2192 | 2192 | 74 | true | true |

## 性能对比

以下为同一次六图 batch 验证中的端到端耗时，包含预处理和 greedy decode。ONNX 行内耗时为 batch 总耗时按样本平均分摊，真实 wall time 见后面的 batch 汇总。`speedup_total = Transformers total / ONNX total_share`，大于 1 表示 ONNX 更快。

| 图片 | Transformers 总耗时(s) | ONNX batch 分摊耗时(s) | speedup_total |
| --- | ---: | ---: | ---: |
| `formula-2026-01-18-152316.png` | 4.774 | 2.020 | 2.363 |
| `formula_2025-8-2_17-28-16.jpg` | 1.357 | 2.020 | 0.672 |
| `table-2026-01-01-202211.png` | 21.606 | 43.394 | 0.498 |
| `table-2026-05-23-124132.png` | 14.455 | 43.394 | 0.333 |
| `text-2026-05-23-124542.png` | 2.657 | 9.215 | 0.288 |
| `text-2026-05-23-124711.png` | 8.320 | 9.215 | 0.903 |

Batch 汇总：

| Batch | 图片 | Padded Prompt | ONNX batch 总耗时(s) |
| ---: | --- | ---: | ---: |
| 1 | `formula-2026-01-18-152316.png`, `formula_2025-8-2_17-28-16.jpg` | 400 | 4.041 |
| 2 | `table-2026-01-01-202211.png`, `table-2026-05-23-124132.png` | 3472 | 86.787 |
| 3 | `text-2026-05-23-124542.png`, `text-2026-05-23-124711.png` | 2192 | 18.430 |

六图合计：

- Transformers 总耗时：53.170s
- ONNX batch token-only unified KV + I/O binding 总耗时：109.258s
- overall speedup_total：0.487
- `falcon_ocr_kv_token.onnx` 文件大小：1,081,568,222 bytes
- 旧的双图方式需要同时加载 `falcon_ocr_prefill.onnx` 和 `falcon_ocr_decode.onnx`，合计约 2,160,763,294 bytes；统一图只加载一份权重。

## 数据传输优化

Falcon-OCR vocab size 为 `65536`。旧 unified KV 图每步需要回传完整 next-token logits：

| 版本 | 每步 token 输出 | 单步 token 输出传输 | 缩减比 |
| --- | --- | ---: | ---: |
| `falcon_ocr_kv.onnx` | `65536` float32 logits | 262,144 bytes | 1x |
| `falcon_ocr_kv_token.onnx` | 1 int64 token ID | 8 bytes | 32,768x |

以本次六图 batch 验证的 ONNX 生成步数计算，实际生成 token 数为 `119 + 31 + 201 + 326 + 68 + 74 = 819`。仅 token 输出部分，完整 logits 回传约 `214.7 MB`，token-only 回传约 `6.6 KB`。KV cache 仍通过 ORT `OrtValue` 在 ORT/CUDA 侧传递，不转回 numpy。

当前 token-only batch 推理已验证数值一致。与上一轮完整 logits 输出图相比，六图 batch ONNX 总耗时从 `203.212s` 降到 `109.258s`。由于 batch 内仍必须 pad 到同组最长 prompt，长度差异大的图片会让短样本跟随最长序列计算，实际部署建议按 prompt/image token 长度分桶后 batch；下一步性能空间主要在 prefill / dense attention ONNX 路径、KV cache 原位更新，以及 ORT fused attention / TensorRT plugin / CUDA graph 等进一步优化。
