
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = r"d:\models\GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": r"D:\tmp\formula-2026028-105537.jpg",
            },
            {
                "type": "text",
                "text": "Formula Recognition:"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    dtype="float32"
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)

print("-"*53)
print(generated_ids[0][inputs["input_ids"].shape[1]:])
