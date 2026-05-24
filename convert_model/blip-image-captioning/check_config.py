from transformers import BlipConfig
MODEL_PATH = r"D:\models\blip-image-captioning-large"
config = BlipConfig.from_pretrained(MODEL_PATH)

print(f"Vision Hidden Size: {config.vision_config.hidden_size}")
print(f"Text Hidden Size: {config.text_config.hidden_size}")
print(f"Projection Dim: {config.projection_dim}")
