from transformers import BlipConfig
MODEL_PATH = r"D:\models\blip-image-captioning-large"
config = BlipConfig.from_pretrained(MODEL_PATH)

print(f"Image Size: {config.vision_config.image_size}")
print(f"Patch Size: {config.vision_config.patch_size}")
