

# PP-FormulaNet Torch → ONNX Converter

将 PaddlePaddle/PP-FormulaNet_plus-L 和 PP-FormulaNet-L 的 PyTorch 权重转换为 ONNX 格式，
并验证推理结果准确度（对比每一步 Top-5 Logits）。

## 项目结构

```
├── run_conversion.py              # 主转换与验证脚本（完整 logits）
├── run_conversion_optimized.py    # 性能优化版（token-only 输出）
├── run_inference_optimized.py     # 优化版推理脚本（最小数据传输）
├── onnx/
│   ├── PP-FormulaNet_plus-L_encoder.onnx       # Plus-L 视觉编码器 (383 MB)
│   ├── PP-FormulaNet_plus-L_decoder.onnx       # Plus-L 文本解码器（完整 logits, 345 MB）
│   ├── PP-FormulaNet_plus-L_decoder_prefill.onnx   # Plus-L KV-cache 预填充
│   ├── PP-FormulaNet_plus-L_decoder_decode.onnx    # Plus-L KV-cache 解码
│   ├── PP-FormulaNet_plus-L_decoder_model_merged.onnx  # Plus-L merged KV 解码器
│   ├── PP-FormulaNet-L_encoder.onnx            # L 视觉编码器 (383 MB)
│   ├── PP-FormulaNet-L_decoder.onnx            # L 文本解码器（完整 logits, 342 MB）
│   ├── PP-FormulaNet-L_decoder_opt.onnx        # L 优化解码器（token-only, ~342 MB）
│   ├── PP-FormulaNet-L_decoder_model_merged.onnx   # L merged KV 解码器
│   ├── PP-FormulaNet_plus-L_verification.json  # Plus-L 验证报告
│   ├── PP-FormulaNet_plus-L_kv_verification.json    # Plus-L KV-cache 验证报告
│   ├── PP-FormulaNet_plus-L_optimized_verification.json  # Plus-L 优化版验证
│   ├── PP-FormulaNet-L_verification.json       # L 验证报告
│   └── PP-FormulaNet-L_kv_verification.json         # L KV-cache 验证报告
├── imgs/
│   └── formula-2026-01-18-152316.png
└── README.md
```

## 环境要求

- Python 3.11 + conda 环境 `transformers5`
- transformers 5.8.0
- PyTorch 2.7.1
- ONNX 1.20.1
- ONNX Runtime 1.24.2
- Pillow

## 使用方法

### 转换并验证所有模型（强制 max_length=1537）

```bash
conda activate transformers5
python run_conversion.py                  # 标准版（完整 logits）
python run_conversion_optimized.py        # 优化版（token-only）
```

`run_conversion_optimized.py` 会强制使用模型 `generation_config.max_length` 作为验证步数；`--steps` 仅作为旧参数保留，不会缩短验证长度。

### 性能优化版：Merged KV-cache 解码器

优化版默认导出并验证 `decoder_model_merged.onnx`。这张 decoder 图同时处理首步 prefill 和后续 decode：首步传入长度为 0 的 self-KV，后续步骤回喂上一轮输出的 self-KV。它只返回 **token ID** 和 16 个 self-KV 张量，不返回完整 logits 或 cross-KV。

```bash
# 转换并验证优化版
python run_conversion_optimized.py --model plus

# 默认批量处理 imgs/ 下的图片
python run_conversion_optimized.py

# 指定单张图片或自定义目录
python run_conversion_optimized.py --image "path/to/image.png"
python run_conversion_optimized.py --image "path/to/images/"

# 仅推理（无需 PyTorch）
python run_inference_optimized.py --model plus --image "path/to/image.png"

# 兼容旧的 split prefill/decode 模型
python run_inference_optimized.py --model plus --image "path/to/image.png" \
  --prefill onnx/PP-FormulaNet_plus-L_decoder_prefill.onnx \
  --decode onnx/PP-FormulaNet_plus-L_decoder_decode.onnx
```

`run_inference_optimized.py` 同样强制使用模型 `generation_config.max_length`；`--max-steps` 仅作为旧参数保留，不会缩短生成长度。

**数据传输对比（287 步生成）：**

| 版本 | 每步输出 | 总数据传输 | 缩减比 |
|------|----------|-----------|--------|
| 标准版 | 50,000 float32 logits | ~57 MB | 1× |
| Merged KV 版 | 1 int64 token ID + self-KV | token 约 2.3 KB + KV cache | **~25,000× token 输出** |

### 仅验证已有 ONNX 文件（全量步骤）

```bash
python run_conversion.py --skip-export
python run_conversion_optimized.py --skip-export
```

### 仅转换指定模型

```bash
python run_conversion.py --model plus    # 仅 PP-FormulaNet_plus-L
python run_conversion.py --model L       # 仅 PP-FormulaNet-L
```

### 强制重新导出

```bash
python run_conversion.py --force
```

### 使用自定义验证图片

```bash
python run_conversion.py --image "path/to/your/image.png"
```

## 模型架构

PP-FormulaNet 是一个视觉到文本的编码器-解码器模型，用于公式识别：

- **编码器 (Vision Encoder)**: ViT-based，包含 window attention 和 relative position embeddings
  - 输入: `pixel_values` [B, 3, 768, 768]
  - 输出: `pooler_output` [B, 144, 512] (经过 multi_modal_projector 投影)
  
- **解码器 (Text Decoder)**: Transformer decoder with cross-attention
  - 输入: `decoder_input_ids` [B, seq_len], `encoder_hidden_states` [B, 144, 512]
  - 输出: `logits` [B, seq_len, 50000]

## 验证结果（max_length=1537，完整序列验证）

### PP-FormulaNet_plus-L（287 步）
| 指标 | 结果 |
|------|------|
| 验证步数 | **287 steps**（完整生成序列） |
| Top-1 匹配率 | **100.00%** (287/287) |
| Top-5 完全匹配率 | **100.00%** (287/287) |
| 最大 Logit 差异 | 3.43e-05 |
| 生成路径匹配 | ✅ 完全一致 |
| 文本输出匹配 | ✅ 完全一致 |
| PyTorch 耗时 | 11.1s |
| ONNX 耗时 | 8.3s |

### 优化版（KV-Cache，token-only 输出）

**正确性验证（max_length=1537，对比 `m.generate()`）：**

| 指标 | Plus-L |
|------|--------|
| 验证步数 | **288 steps**（与 `m.generate()` 完全一致） |
| Token 匹配率 | **100.00%** (288/288) |
| 生成路径匹配 | ✅ 完全一致 |
| 文本输出匹配 | ✅ 完全一致 |
| 公式 | `w(z) = \frac{\hbar}{2\pi^2} \int_0^\infty dk_z \int_0^\infty dk_\parallel ...` |

**性能与显存：**

| 指标 | 原版 | KV-Cache 优化版 |
|------|------|-----------------|
| Token 输出/步 | 200 KB (50000 logits) | **8 bytes** (1 token id) |
| Decoder ONNX session | 1 个 O(n²) decoder | **1 个 merged O(n) decoder** |
| Prefill/decode 权重 | 单图全序列 | **不同时加载 split 两份 decoder 权重** |
| KV 输出 | 无 | **16 个 self-KV，不返回 cross-KV** |
| PyTorch 耗时 | 11.1s | 4.2s |
| ONNX 耗时 | 8.3s | 6.0s |
| 复杂度 | O(n²) | **O(n)** |

> **说明**：merged 解码器的 cross-attention 每步从 encoder_hidden_states 重新计算（不缓存 cross-KV），确保与 `m.generate()` 输出完全一致。self-attention 使用 KV-cache 实现 O(n) 复杂度。旧的 `decoder_prefill.onnx` 和 `decoder_decode.onnx` 仍保留，便于回归或兼容已有调用方。

### PP-FormulaNet-L（273 步）
| 指标 | 结果 |
|------|------|
| 验证步数 | **273 steps**（完整生成序列） |
| Top-1 匹配率 | **100.00%** (273/273) |
| Top-5 完全匹配率 | **100.00%** (273/273) |
| 最大 Logit 差异 | 2.43e-05 |
| 生成路径匹配 | ✅ 完全一致 |
| 文本输出匹配 | ✅ 完全一致 |
| PyTorch 耗时 | 10.6s |
| ONNX 耗时 | 7.8s |

## 技术说明

1. **ONNX 导出方式**: 使用经典 Torch JIT Tracer (非 dynamo)
2. **动态轴支持**: 编码器支持动态 batch_size；merged 解码器支持动态 batch_size、decoder_seq_len、encoder_seq_len 和 past/present self-KV 长度
3. **Masking 处理**: 通过 monkey-patch `create_causal_mask` 返回 None，让 SDPA 使用内置的 `is_causal=True` 机制
4. **Merged decoder 接口**: 输入 `ids`、`enc` 和 16 个 `past_{layer}_{sk,sv}`；输出 `tok` 和 16 个 `pres_{layer}_{sk,sv}`。首步 past self-KV 形状为 `[B, heads, 0, head_dim]`
5. **验证方法**: PyTorch 和 ONNX 都逐步生成 token，并对比完整 token 序列、EOS 状态和解码文本
6. **验证覆盖**: 强制使用 `generation_config.max_length=1537`，即使某些样本不能停止也会跑到模型最大长度，用于暴露后续步骤错误

## 验证图片

默认使用 `imgs/` 目录中的图片，按文件名排序后批量验证。
