# 将PP-DocLayoutV3模型转换为ONNX

## 环境
使用conda的paddlex环境，需安装以下依赖：

```bash
conda activate paddlex
pip install onnx onnxruntime transformers safetensors
```

> 注意：需要 transformers >= 5.8 以支持 `pp_doclayout_v3` 架构。

## 目录
- torch目录：`D:\models\PP-DocLayoutV3_safetensors`
- ONNX目录：`D:\models\layout\PP-DocLayoutV3`
- ONNX文件：`D:\models\layout\PP-DocLayoutV3\model.onnx`

## 约束
1. 必须使用`imgs`文件夹下的图片批量推理验证正确性
2. 必须支持批量推理，即输入为一个batch的图片，输出为一个batch的结果
3. 必须支持动态输入，即输入的图片大小可以不一致

## 使用方法

```bash
conda activate paddlex

# 转换 + 验证（默认）
python convert.py

# 仅导出 ONNX
python convert.py --export

# 仅验证
python convert.py --verify
```

## ONNX 模型信息

| 项目 | 说明 |
|------|------|
| 输入 | `pixel_values` - float32, shape `(batch, 3, H, W)` |
| 输出1 | `logits` - float32, shape `(batch, 300, 25)` — 300 queries, 25类别分类logits |
| 输出2 | `pred_boxes` - float32, shape `(batch, 300, 4)` — 归一化 cx, cy, w, h |
| 输出3 | `order_logits` - float32, shape `(batch, 300, 300)` — 阅读顺序logits |
| Opset | 17 |
| 动态轴 | batch, height, width |

## 后处理说明

脚本内置了简化版后处理函数 `post_process()`，逻辑与 transformers 官方一致但不含 mask/polygon：
1. `order_logits` → sigmoid → argmax 得到阅读顺序
2. `pred_boxes` 从 cxcywh 转换为 xyxy 并缩放到原图尺寸
3. `logits` → sigmoid → topk 过滤低于阈值的检测框
4. 按阅读顺序排序输出

## 验证结果

2张测试图片全部通过，PyTorch 与 ONNX 推理结果一致：
- 检测框数量完全一致
- 标签完全一致
- 最大分数差异 < 0.000001
- 最大坐标差异 < 0.0002
