"""
Validate ONNX model with real images
"""
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os

model_path = 'D:/models/GLM-OCR'
onnx_path = 'D:/models/onnx/GLM-OCR-LLM'

print('Loading model and processor...')
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float32
)
model.eval()

# Load ONNX session
print('Loading ONNX session...')
prefill_session = ort.InferenceSession(
    f'{onnx_path}/llm_prefill.onnx', 
    providers=['CPUExecutionProvider']
)

real_images = [
    'd:/tmp/table-2026-01-01-202211.png',
    'd:/tmp/formula_2025-8-2_17-28-16.jpg'
]

print()
print('=== Validating with real images ===')
for img_path in real_images:
    if not os.path.exists(img_path):
        print(f'  [SKIP] {img_path} not found')
        continue
        
    try:
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        print(f'  {filename}: size={image.size}')
        
        messages = [{
            'role': 'user', 
            'content': [
                {'type': 'image'}, 
                {'type': 'text', 'text': 'Extract text from this image.'}
            ]
        }]
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text_input, images=[image], return_tensors='pt')
        
        # Get visual hidden states using the visual encoder
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']
        
        # Call visual model to get image embeds
        with torch.no_grad():
            visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
            # visual_output is the actual hidden states tensor
            if hasattr(visual_output, 'last_hidden_state'):
                image_embeds = visual_output.last_hidden_state
            else:
                # It's already the tensor
                image_embeds = visual_output
        
        # Get input embeddings (combines text + image)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            # Manual embedding computation - embed_tokens is directly on language_model
            embed_tokens = model.language_model.embed_tokens
            text_embeds = embed_tokens(input_ids)
            
            # Find image token positions and merge
            image_token_id = model.config.image_token_id
            image_mask = input_ids == image_token_id
            
            # Replace image token positions with image embeddings
            num_image_tokens = image_mask.sum().item()
            print(f'    Input tokens: {input_ids.shape[1]}, Image tokens: {num_image_tokens}')
            
            # Create final inputs_embeds
            inputs_embeds = text_embeds.clone()
            if num_image_tokens > 0 and image_embeds.shape[1] == num_image_tokens:
                inputs_embeds[image_mask] = image_embeds.view(-1, image_embeds.shape[-1])
            
            # Get position_ids from inputs or create default
            position_ids = inputs.get('position_ids')
            if position_ids is None:
                seq_len = inputs_embeds.shape[1]
                batch_size = 1
                position_ids = torch.zeros((3, batch_size, seq_len), dtype=torch.long)
                for i in range(3):
                    position_ids[i] = torch.arange(seq_len).unsqueeze(0)
            
            # Ensure position_ids has shape [3, batch, seq]
            if position_ids.shape[0] != 3:
                position_ids = position_ids.permute(1, 0, 2)
            
            # PyTorch forward
            pt_output = model.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=False
            )
            
            # Load lm_head weights (since AutoModel doesn't include it)
            import safetensors.torch
            lm_head_weight = None
            for sf_file in ['model.safetensors', 'model-00001-of-00001.safetensors']:
                sf_path = os.path.join(model_path, sf_file)
                if os.path.exists(sf_path):
                    tensors = safetensors.torch.load_file(sf_path)
                    if 'lm_head.weight' in tensors:
                        lm_head_weight = tensors['lm_head.weight'].float()  # Convert to float32
                        break
            
            if lm_head_weight is not None:
                pt_logits = torch.nn.functional.linear(pt_output.last_hidden_state.float(), lm_head_weight)
            else:
                # Use tied weights from embed_tokens if lm_head not found
                pt_logits = torch.nn.functional.linear(pt_output.last_hidden_state.float(), embed_tokens.weight.float())
            
            # ONNX forward
            inputs_embeds_np = inputs_embeds.numpy().astype(np.float32)
            pos_ids_np = position_ids.numpy().astype(np.int64)
            
            # Create attention mask - all ones for valid tokens
            batch_size, seq_len = inputs_embeds.shape[:2]
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            
            onnx_outputs = prefill_session.run(None, {
                'inputs_embeds': inputs_embeds_np,
                'position_ids': pos_ids_np,
                'attention_mask': attention_mask
            })
            onnx_logits = onnx_outputs[0]
            
            # Compare
            max_diff = np.max(np.abs(pt_logits.numpy() - onnx_logits))
            status = '[PASS]' if max_diff < 1e-3 else '[FAIL]'
            print(f'    {status} max_diff={max_diff:.2e}')
            
    except Exception as e:
        import traceback
        print(f'    [ERROR] {e}')
        traceback.print_exc()

print()
print('=== Validation Complete ===')
