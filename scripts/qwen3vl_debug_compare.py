"""
Debug comparison: torch vs ONNX intermediate values.
Prints torch position_ids, pixel_values, and image_features to compare with ONNX.
"""
import torch
import numpy as np
import math
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_DIR = r"D:\models\Qwen3-VL-2B-Instruct"
IMAGE_PATH = r"D:\tmp\formula_2025-8-2_17-28-16.jpg"

# ------ ONNX preprocessing (copied from qwen3vl_onnx_infer.py) ------
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(max_pixels / (height * width))
        h_bar = math.floor(height * beta / factor) * factor
        w_bar = math.floor(width * beta / factor) * factor
    return h_bar, w_bar

def preprocess_image_onnx(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    new_h, new_w = smart_resize(orig_h, orig_w)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    print(f"  ONNX Image Resize: {orig_w}x{orig_h} -> {new_w}x{new_h}")
    pixels = np.array(img, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    pixels = pixels.transpose(2, 0, 1)  # [3, H, W]
    pixels = np.stack([pixels, pixels], axis=1)  # [3, 2, H, W]
    C, T, H, W = pixels.shape
    grid_t = T // TEMPORAL_PATCH_SIZE
    grid_h = H // PATCH_SIZE
    grid_w = W // PATCH_SIZE
    patches = pixels.reshape(C, grid_t, TEMPORAL_PATCH_SIZE, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE)
    patches = patches.transpose(1, 3, 5, 0, 2, 4, 6)
    num_patches = grid_t * grid_h * grid_w
    patch_dim = C * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    pixel_values = patches.reshape(num_patches, patch_dim).astype(np.float32)
    image_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, image_grid_thw


def main():
    print("=" * 60)
    print("STEP 1: Compare preprocessing")
    print("=" * 60)

    # Torch preprocessing
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "url": IMAGE_PATH},
            {"type": "text", "text": "Convert this formula image to LaTeX.\n/no_think"},
        ]},
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    torch_pv = inputs["pixel_values"].numpy()
    torch_grid = inputs["image_grid_thw"].numpy()
    torch_input_ids = inputs["input_ids"][0].numpy()

    print(f"  Torch pixel_values: {torch_pv.shape}, dtype={torch_pv.dtype}")
    print(f"  Torch image_grid_thw: {torch_grid}")
    print(f"  Torch input_ids len: {len(torch_input_ids)}")
    print(f"  Torch input_ids: {torch_input_ids.tolist()}")

    # ONNX preprocessing
    onnx_pv, onnx_grid = preprocess_image_onnx(IMAGE_PATH)
    print(f"  ONNX pixel_values: {onnx_pv.shape}")
    print(f"  ONNX image_grid_thw: {onnx_grid}")

    # Compare pixel values
    if torch_pv.shape == onnx_pv.shape:
        diff = np.abs(torch_pv - onnx_pv)
        print(f"  Pixel values - max diff: {diff.max():.8f}, mean diff: {diff.mean():.8f}")
        print(f"  Pixel values match: {np.allclose(torch_pv, onnx_pv, atol=1e-4)}")
    else:
        print(f"  SHAPE MISMATCH! torch={torch_pv.shape} vs onnx={onnx_pv.shape}")

    # Compare pixel_values sample
    print(f"  Torch pv[0,:5]: {torch_pv[0,:5]}")
    print(f"  ONNX  pv[0,:5]: {onnx_pv[0,:5]}")

    # Compare grids
    print(f"  Grid match: {np.array_equal(torch_grid, onnx_grid)}")

    print("\n" + "=" * 60)
    print("STEP 2: Compare position_ids")
    print("=" * 60)

    # Get torch position_ids via model's get_rope_index
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    )
    model.eval()

    with torch.no_grad():
        rope_result = model.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
        )
    torch_pos_ids = rope_result[0].numpy()  # position_ids
    print(f"  Torch position_ids shape: {torch_pos_ids.shape}")
    print(f"  Torch position_ids[0,0,:20]: {torch_pos_ids[0,0,:20]}")
    print(f"  Torch position_ids[1,0,:20]: {torch_pos_ids[1,0,:20]}")
    print(f"  Torch position_ids[2,0,:20]: {torch_pos_ids[2,0,:20]}")
    
    # Print around image region
    IMAGE_PAD = 151655
    pad_positions = np.where(torch_input_ids == IMAGE_PAD)[0]
    if len(pad_positions) > 0:
        img_start = pad_positions[0]
        img_end = pad_positions[-1] + 1
        print(f"  Image pad positions: {img_start} to {img_end-1}")
        print(f"  Torch pos_ids[0] around image start: {torch_pos_ids[0,0,img_start-2:img_start+5]}")
        print(f"  Torch pos_ids[1] around image start: {torch_pos_ids[1,0,img_start-2:img_start+5]}")
        print(f"  Torch pos_ids[2] around image start: {torch_pos_ids[2,0,img_start-2:img_start+5]}")
        print(f"  Torch pos_ids[0] around image end: {torch_pos_ids[0,0,img_end-2:img_end+3]}")
        print(f"  Torch pos_ids[1] around image end: {torch_pos_ids[1,0,img_end-2:img_end+3]}")
        print(f"  Torch pos_ids[2] around image end: {torch_pos_ids[2,0,img_end-2:img_end+3]}")
    
    # Print full dim-0 position IDs
    print(f"\n  FULL Torch position_ids[0,0,:]: {torch_pos_ids[0,0,:].tolist()}")
    print(f"  FULL Torch position_ids[1,0,:]: {torch_pos_ids[1,0,:].tolist()}")
    print(f"  FULL Torch position_ids[2,0,:]: {torch_pos_ids[2,0,:].tolist()}")

    print("\n" + "=" * 60)
    print("STEP 3: Compare vision encoder output")
    print("=" * 60)

    # Run torch vision encoder
    with torch.no_grad():
        visual = model.model.visual
        pv_tensor = torch.tensor(torch_pv)
        grid_tensor = torch.tensor(torch_grid)
        torch_image_features = visual(pv_tensor, grid_thw=grid_tensor)
        print(f"  Torch image_features shape: {torch_image_features.shape}")  # [num_merged, hidden]
        print(f"  Torch image_features[0,:5]: {torch_image_features[0,:5].numpy()}")
        print(f"  Torch image_features[-1,:5]: {torch_image_features[-1,:5].numpy()}")
        print(f"  Torch image_features stats: min={torch_image_features.min():.4f}, max={torch_image_features.max():.4f}, mean={torch_image_features.mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
