
## Qwen3-VL-2B-Instruct

### Overview

Qwen3-VL-2B-Instruct is a vision-language model for image understanding and OCR tasks. This implementation provides a complete pipeline: PyTorch reference inference → ONNX export → Python ONNX inference → Java ONNX inference, all validated to produce identical outputs.

### Architecture

| Component | Details |
|:----------|:--------|
| Vision Encoder | 24-layer ViT, hidden=1024, patch_size=16, DeepStack at layers 5, 11, 17 |
| PatchMerger | spatial_merge_size=2, projects 4×1024 → 2048 |
| Text Decoder | 28-layer Qwen3, hidden=2048, 16 attn heads, 8 KV heads (GQA), head_dim=128, SwiGLU MLP |
| Vocab Size | 151936 |
| Position Encoding | MRoPE (3D: T/H/W), mrope_section=[24, 20, 20], interleaved |
| DeepStack | 3 levels (layers 5, 11, 17), features added to decoder hidden states |
| Weight Tying | lm_head.weight == embed_tokens.weight |

### ONNX Models

| Model | Size | Inputs | Outputs |
|:------|:-----|:-------|:--------|
| vision_encoder.onnx | 1553 MB | pixel_values [N, 1536], image_grid_thw [I, 3] | image_features [M, 2048], deepstack_0/1/2 [M, 2048] |
| embed_tokens.onnx | 1187 MB | input_ids [B, S] int64 | inputs_embeds [B, S, 2048] |
| decoder_model_merged.onnx | 6.56 GB | inputs_embeds [B, S, 2048], attention_mask [B, T], position_ids [3, B, S], deepstack_scattered_0/1/2 [B, S, 2048], past_key_values.{0-27}.key/value [B, 8, P, 128] | logits [B, S, 151936], present.{0-27}.key/value [B, 8, T, 128] |

### ONNX Export Design: Pre-Scattered DeepStack

The original PyTorch DeepStack implementation uses boolean mask indexing (`hidden_states[visual_pos_masks, :] += visual_embeds`), which cannot be exported dynamically via `torch.onnx.export`. The solution:

1. **Pre-scatter on the caller side**: Instead of passing compact `[num_vis, hidden]` tensors + a boolean mask, pass `[batch, seq, hidden]` tensors with vision features already placed at the correct positions and zeros elsewhere.
2. **Monkey-patch `_deepstack_process`**: Replace the boolean-indexed accumulation with simple element-wise addition (`hidden_states = hidden_states + deepstack_scattered`).
3. **Decode steps**: Pass all-zeros deepstack tensors (no vision features to add).

### Image Preprocessing

Matches the transformers `Qwen2VLImageProcessorFast._preprocess` pipeline:

1. **smart_resize**: Resize to multiples of `factor=32` (patch_size × merge_size), constrained by `min_pixels=65536` and `max_pixels=16777216`
2. **Normalize**: `(pixel / 255 - 0.5) / 0.5`
3. **Temporal duplication**: Stack 2 identical frames for `temporal_patch_size=2`
4. **Patch extraction with merge_size grouping**:
   - Reshape to `[grid_t, temp, C, gh/merge, merge_h, patch_h, gw/merge, merge_w, patch_w]`
   - Permute to `[grid_t, gh/merge, gw/merge, merge_h, merge_w, C, temp, patch_h, patch_w]`
   - Flatten to `[num_patches, 1536]`

### MRoPE Position IDs

Position IDs are 3-dimensional `[3, batch, seq]` for T/H/W:

- **Text tokens**: All 3 dimensions share the same incrementing position
- **Image tokens**: T = constant offset per temporal frame, H = row index in merged grid, W = column index in merged grid
- **Key rule**: `st_idx` is computed once per text+vision iteration (not separately for text and vision segments)

### Issues Fixed During Implementation

#### 1. SDPA GQA Assertion Error during ONNX Export

**Problem**: `torch.onnx.export` failed with `AssertionError` in SDPA attention when using Grouped Query Attention (GQA, 16 query heads vs 8 KV heads).

**Fix**: Set `attn_implementation="eager"` when loading the model for export, bypassing the SDPA kernel.

#### 2. `DynamicCache.from_legacy_cache` Removed in Transformers 5.x

**Problem**: The ONNX export script used the legacy tuple-based KV-cache API (`DynamicCache.from_legacy_cache`), which was removed in transformers 5.0.

**Fix**: Use the new `DynamicCache()` API with `.update(key, value, layer_idx)` for construction and `.layers[i].keys/.values` for access.

#### 3. Boolean Indexing in DeepStack Cannot Export Dynamically

**Problem**: `hidden_states[visual_pos_masks, :] += visual_embeds` uses boolean indexing that becomes static during ONNX tracing, baking in a fixed number of visual tokens.

**Fix**: Redesigned to pre-scattered deepstack tensors `[batch, seq, hidden]` as explicit model inputs, with a monkey-patched `_deepstack_process` that uses simple element-wise addition. See "Pre-Scattered DeepStack" section above.

#### 4. Incorrect Patch Extraction Order (Preprocessing Bug)

**Problem**: The ONNX inference script used a naive row-major patch extraction (`[C, grid_t, temp, grid_h, patch, grid_w, patch]` → transpose `[grid_t, grid_h, grid_w, C, temp, patch, patch]`), which doesn't match the transformers processor that groups patches by `merge_size` before flattening.

**Symptom**: pixel_values had `max_diff=2.0` vs transformers reference, causing completely wrong decoder output (4/31 tokens matched).

**Fix**: Match the transformers reshape+permute exactly:
```
[grid_t, temp, C, gh/merge, merge_h, patch_h, gw/merge, merge_w, patch_w]
  → permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
  → [grid_t, gh/merge, gw/merge, merge_h, merge_w, C, temp, patch_h, patch_w]
  → flatten to [num_patches, 1536]
```

#### 5. Position IDs `st_idx` Computed Incorrectly

**Problem**: In the MRoPE position_ids computation, `st_idx` was computed separately for the text segment and the vision segment within the same iteration. The transformers `get_rope_index` computes `st_idx` once before both segments.

**Symptom**: Image token positions started at `30` instead of `15`, causing position mismatch and wrong decoder output.

**Fix**: Compute `st_idx` once at the beginning of each iteration, before processing both text and vision segments:
```python
# BEFORE (bug): st_idx computed twice
if text_len > 0:
    st_idx = ... # computed here
    llm_pos_ids_list.append(text_pos + st_idx)
st_idx = ... # computed again (different value!)
llm_pos_ids_list.append(vis_pos + st_idx)

# AFTER (fix): st_idx computed once
st_idx = ... # computed once
if text_len > 0:
    llm_pos_ids_list.append(text_pos + st_idx)
llm_pos_ids_list.append(vis_pos + text_len + st_idx)
```

#### 6. ONNX External Data File Duplication

**Problem**: When consolidating ONNX external data, the old `.data` file was not deleted before re-saving, causing the new data to be appended to the old file (doubling the file size).

**Fix**: Clean export by deleting old data files before `onnx.save_model()` with `convert_attribute=True`.

### Validation Results

| Comparison | Result |
|:-----------|:-------|
| Torch → Python ONNX | 31/31 tokens PERFECT MATCH |
| Python ONNX → Java ONNX | 31/31 tokens PERFECT MATCH |

**Test image**: `D:\tmp\formula_2025-8-2_17-28-16.jpg` (389×173 → 384×192 after smart_resize)

**Output**: `n! \approx \sqrt{2\pi n} \left( \frac{n}{e} \right)^n`

**Token IDs**: `[73594, 64680, 198, 77, 0, 1124, 48053, 1124, 26888, 90, 17, 59, 2493, 308, 92, 1124, 2359, 7, 1124, 37018, 91362, 15170, 68, 92, 1124, 1291, 29776, 77, 198, 73594, 151645]`

### Files

| File | Purpose |
|:-----|:--------|
| `scripts/qwen3vl_torch_infer.py` | PyTorch reference inference (ground truth) |
| `convert_model/export_qwen3vl_to_onnx_v2.py` | ONNX export with pre-scattered deepstack |
| `scripts/qwen3vl_onnx_infer.py` | Python ONNX inference with KV-cache |
| `scripts/inspect_qwen3vl.py` | ONNX model inspection utility |
| `flux-ocr/.../qwen3vl/Qwen3VlImageProcessor.java` | Image preprocessing (smart_resize + merge-grouped patches) |
| `flux-ocr/.../qwen3vl/Qwen3VlEncoderModel.java` | Vision encoder wrapper (image_features + deepstack) |
| `flux-ocr/.../qwen3vl/Qwen3VlEmbedModel.java` | Token embedding wrapper |
| `flux-ocr/.../qwen3vl/Qwen3VlDecoderModel.java` | Autoregressive decoder with KV-cache + MRoPE + deepstack |
| `flux-ocr/.../qwen3vl/Qwen3VlModel.java` | Pipeline orchestrator |
| `flux-ocr/.../qwen3vl/Qwen3VlModelTest.java` | Test with ground truth validation |

### Code Example

```java
static void main() throws Exception {
    String modelRootDir = "D:\\models\\onnx";
    String modelName = "Qwen3-VL-2B-Instruct";
    String imagePath = "D:\\tmp\\formula_2025-8-2_17-28-16.jpg";

    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
         MatManager matManager = new MatManager();
         NDManager ndManager = NDManager.newBaseManager()) {

        Qwen3VlModel model = new Qwen3VlModel(modelRootDir, modelName, -1, env);

        Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
        PreProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);

        List<TextResult> results = model.doBatchPredict(
                List.of(ppr), matManager, ndManager, null);

        for (TextResult result : results) {
            System.out.println(result.text());
        }
        model.close();
    }
}
```

### Known Limitations

- **Vision encoder tracing**: The ONNX vision encoder has internal `cu_seqlens` computation that bakes in the traced grid size. Currently validated for images producing grid `[1, 12, 24]`. Different image sizes may produce incorrect results and would need re-export or restructuring.
- **Batch size**: Currently only supports batch size 1 due to the input_ids/position_ids construction being tied to a single image.

