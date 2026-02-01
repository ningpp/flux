"""
Script to convert CodeFormulaV2 (Idefics3Model) to ONNX format.
Following the same pattern as granite-docling-258M-ONNX.

Exports:
1. vision_encoder.onnx - Vision model (with multi-image support)
2. embed_tokens.onnx - Text embedding layer
3. decoder_model_merged.onnx - Text decoder with KV cache
"""

import torch
import torch.onnx
from transformers import AutoModel, AutoProcessor, AutoConfig
from transformers.cache_utils import DynamicCache
from pathlib import Path
import warnings


class CodeFormulaV2ONNXConverter:
    def __init__(self, model_id="docling-project/CodeFormulaV2"):
        """Initialize the converter with the model ID."""
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load_model(self):
        """Load the CodeFormulaV2 model and processor."""
        print(f"Loading model from {self.model_id}...")
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for ONNX compatibility
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model.eval()
        print("✓ Model loaded successfully!")

        # Print model configuration
        print("\nModel Configuration:")
        print(f"  Vision hidden size: {self.config.vision_config.hidden_size}")
        print(f"  Text hidden size: {self.config. text_config.hidden_size}")
        print(f"  Text vocab size: {self.config.text_config.vocab_size}")
        print(f"  Num hidden layers: {self.config.text_config.num_hidden_layers}")
        print(f"  Num attention heads: {self.config.text_config.num_attention_heads}")
        print(f"  Num key-value heads: {self.config.text_config. num_key_value_heads}")
        print(f"  Head dim: {self.config.text_config.head_dim if hasattr(self.config.text_config, 'head_dim') else self.config.text_config.hidden_size // self.config.text_config. num_attention_heads}")
        print(f"  Scale factor: {self.config.scale_factor}")
        print(f"  Image token id: {self.config. image_token_id}")

    def export_vision_encoder(self, output_path="vision_encoder. onnx"):
        """
        Export vision encoder that handles multi-image inputs.
        Input:  pixel_values (batch, num_images, channels, height, width)
               pixel_attention_mask (batch, num_images, height, width) - BOOL type
        Output: image_features (batch, num_images, reduced_seq_len, projected_hidden_size)
        """
        print("\n=== Exporting Vision Encoder ===")

        vision_model = self.model.vision_model
        connector = self.model.connector
        vision_model.eval()
        connector.eval()

        # Get dimensions
        image_size = self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        scale_factor = self.config.scale_factor

        # Wrapper that combines vision model + connector
        class VisionEncoderWrapper(torch.nn.Module):
            def __init__(self, vision_model, connector, patch_size, image_size):
                super().__init__()
                self.vision_model = vision_model
                self.connector = connector
                self.patch_size = patch_size
                self.image_size = image_size

            def forward(self, pixel_values, pixel_attention_mask):
                # Handle both 4D and 5D inputs
                original_shape = pixel_values.shape

                if len(original_shape) == 5:
                    # Multi-image:  (batch, num_images, channels, height, width)
                    batch_size, num_images = original_shape[0], original_shape[1]
                    # Flatten to (batch * num_images, channels, height, width)
                    pixel_values = pixel_values.reshape(-1, *original_shape[2:])
                    pixel_attention_mask = pixel_attention_mask.reshape(-1, *pixel_attention_mask.shape[2:])
                else:
                    # Single image:  (batch, channels, height, width)
                    batch_size, num_images = original_shape[0], 1

                # Convert pixel_attention_mask to patch_attention_mask
                # pixel_attention_mask:  (batch*num_images, height, width) - BOOL
                # patch_attention_mask:  (batch*num_images, num_patches_h, num_patches_w) - BOOL
                if pixel_attention_mask is not None:
                    # Reshape to (batch*num_images, height, width)
                    b, h, w = pixel_attention_mask.shape
                    num_patches_h = h // self.patch_size
                    num_patches_w = w // self.patch_size

                    # 先转换为 float 进行计算，然后再转回 bool
                    pixel_attention_mask_float = pixel_attention_mask.float()

                    # Reshape and average over each patch
                    patch_attention_mask_float = pixel_attention_mask_float.reshape(
                        b, num_patches_h, self.patch_size, num_patches_w, self.patch_size
                    )
                    # Take mean over patch dimensions and convert to bool
                    # (b, num_patches_h, patch_size, num_patches_w, patch_size) -> (b, num_patches_h, num_patches_w)
                    patch_attention_mask = patch_attention_mask_float.mean(dim=(2, 4)) > 0.5
                else:
                    patch_attention_mask = None

                # Process through vision model
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    patch_attention_mask=patch_attention_mask
                )

                # Apply connector (modality projection + pixel shuffle)
                image_features = self.connector(vision_outputs.last_hidden_state)

                # Reshape back if multi-image
                if len(original_shape) == 5:
                    seq_len = image_features.shape[1]
                    hidden_size = image_features.shape[2]
                    image_features = image_features.reshape(batch_size, num_images, seq_len, hidden_size)

                return image_features

        wrapper = VisionEncoderWrapper(vision_model, connector, patch_size, image_size)
        wrapper.eval()

        # Create dummy inputs (5D for multi-image support)
        batch_size = 1
        num_images = 5
        channels = 3
        dummy_pixel_values = torch.randn(batch_size, num_images, channels, image_size, image_size)

        # ⚠️ 修改:  使用 bool 类型，与实际使用一致
        dummy_pixel_attention_mask = torch.ones(batch_size, num_images, image_size, image_size, dtype=torch.bool)

        print(f"  Input pixel_values shape: {dummy_pixel_values.shape}")
        print(f"  Input pixel_attention_mask shape: {dummy_pixel_attention_mask.shape}")
        print(f"  Input pixel_attention_mask dtype: {dummy_pixel_attention_mask.dtype}")

        # Test forward pass
        with torch.no_grad():
            test_output = wrapper(dummy_pixel_values, dummy_pixel_attention_mask)
            print(f"  Output image_features shape: {test_output.shape}")

        # Export to ONNX
        print(f"Exporting vision encoder to {output_path}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                wrapper,
                (dummy_pixel_values, dummy_pixel_attention_mask),
                output_path,
                input_names=['pixel_values', 'pixel_attention_mask'],
                output_names=['image_features'],
                dynamic_axes={
                    'pixel_values': {0: 'batch_size', 1: 'num_images'},
                    'pixel_attention_mask': {0: 'batch_size', 1: 'num_images'},
                    'image_features': {0: 'batch_size', 1: 'num_images'}
                },
                opset_version=17,
                do_constant_folding=True,
                verbose=False
            )
        print(f"✓ Vision encoder exported successfully")

        # 验证导出的模型输入类型
        try:
            import onnx
            from onnx import TensorProto

            model = onnx.load(output_path)
            print("\n  Verifying input types:")
            for inp in model.graph.input:
                type_name = {
                    TensorProto.FLOAT: "float32",
                    TensorProto.BOOL: "bool",
                    TensorProto.INT64: "int64",
                }.get(inp.type.tensor_type.elem_type, f"unknown({inp.type.tensor_type.elem_type})")
                print(f"    {inp.name}: {type_name}")
        except ImportError:
            print("  (onnx package not available for type verification)")
            raise

    def export_embed_tokens(self, output_path="embed_tokens.onnx"):
        """
        Export text embedding layer.
        Input: input_ids (batch, seq_len)
        Output: embeddings (batch, seq_len, hidden_size)
        """
        print("\n=== Exporting Embed Tokens ===")

        embeddings = self.model.text_model.get_input_embeddings()
        embeddings.eval()

        class EmbedWrapper(torch.nn.Module):
            def __init__(self, embed_layer):
                super().__init__()
                self.embed_layer = embed_layer

            def forward(self, input_ids):
                return self.embed_layer(input_ids)

        wrapper = EmbedWrapper(embeddings)
        wrapper.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 512
        dummy_input_ids = torch.randint(0, self.config.text_config.vocab_size, (batch_size, seq_length))

        print(f"  Input input_ids shape: {dummy_input_ids.shape}")

        # Test forward pass
        with torch.no_grad():
            test_output = wrapper(dummy_input_ids)
            print(f"  Output embeddings shape: {test_output.shape}")

        # Export to ONNX
        print(f"Exporting embed tokens to {output_path}...")
        torch.onnx.export(
            wrapper,
            dummy_input_ids,
            output_path,
            input_names=['input_ids'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'embeddings': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )
        print(f"✓ Embed tokens exported successfully")

    def export_decoder_model_merged(self, output_path="decoder_model_merged.onnx"):
        """
        Export text decoder with KV cache support.
        Inputs:
            - inputs_embeds (batch, seq_len, hidden_size)
            - attention_mask (batch, total_seq_len)
            - past_key_values (per layer:  key/value tensors)
        Outputs:
            - logits (batch, seq_len, vocab_size)
            - present_key_values (updated key/value tensors)
        """
        print("\n=== Exporting Decoder Model (Merged with KV Cache) ===")

        text_model = self.model.text_model
        text_model.eval()
        text_model.config.use_cache = True

        # Get config values
        num_hidden_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        hidden_size = self.config.text_config.hidden_size
        num_attention_heads = self.config.text_config.num_attention_heads
        head_dim = hidden_size // num_attention_heads

        if hasattr(self.config.text_config, 'head_dim'):
            head_dim = self.config.text_config.head_dim

        print(f"  Num hidden layers: {num_hidden_layers}")
        print(f"  Num key-value heads:  {num_key_value_heads}")
        print(f"  Head dim: {head_dim}")

        class DecoderWrapper(torch.nn.Module):
            def __init__(self, text_model, num_layers, num_kv_heads, head_dim):
                super().__init__()
                self.text_model = text_model
                self.num_layers = num_layers
                self.num_kv_heads = num_kv_heads
                self.head_dim = head_dim

            def forward(self, inputs_embeds, attention_mask, *past_key_values_flat):
                # Reorganize flat past_key_values into DynamicCache
                past_cache = None
                if len(past_key_values_flat) > 0 and past_key_values_flat[0].shape[2] > 0:
                    # Only create cache if there's actual past data
                    past_cache = DynamicCache()
                    for i in range(self.num_layers):
                        key = past_key_values_flat[i * 2]
                        value = past_key_values_flat[i * 2 + 1]
                        past_cache.update(key, value, layer_idx=i)

                # Run the model
                outputs = self.text_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_cache,
                    use_cache=True,
                    return_dict=True
                )

                # Get logits (add LM head if needed)
                hidden_states = outputs.last_hidden_state

                # Apply LM head if available
                if hasattr(self.text_model, 'lm_head'):
                    logits = self.text_model.lm_head(hidden_states)
                else:
                    # For base model, use embedding layer in reverse (weight tying)
                    embed_weight = self.text_model.get_input_embeddings().weight
                    logits = torch.matmul(hidden_states, embed_weight.t())

                # Flatten present key values
                present_kvs = []
                for layer_idx in range(self.num_layers):
                    key, value = outputs.past_key_values[layer_idx]
                    present_kvs.append(key)
                    present_kvs.append(value)

                return (logits, *present_kvs)

        wrapper = DecoderWrapper(text_model, num_hidden_layers, num_key_value_heads, head_dim)
        wrapper.eval()

        # Create dummy inputs
        batch_size = 2
        seq_len = 1
        past_seq_len = 4

        dummy_inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        dummy_attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.long)

        # Create past key values
        dummy_past_kvs = tuple(
            torch.randn(batch_size, num_key_value_heads, past_seq_len, head_dim)
            for _ in range(num_hidden_layers * 2)
        )

        print(f"  Input inputs_embeds shape:  {dummy_inputs_embeds.shape}")
        print(f"  Input attention_mask shape: {dummy_attention_mask.shape}")
        print(f"  Number of past_key_value tensors:  {len(dummy_past_kvs)}")
        print(f"  Past key/value shape:  {dummy_past_kvs[0].shape}")

        # Test forward pass
        print("  Testing forward pass...")
        with torch.no_grad():
            test_outputs = wrapper(dummy_inputs_embeds, dummy_attention_mask, *dummy_past_kvs)
            print(f"  ✓ Output logits shape: {test_outputs[0].shape}")
            print(f"  ✓ Number of present_key_value tensors:  {len(test_outputs) - 1}")
            print(f"  ✓ Present key/value shape: {test_outputs[1].shape}")

        # Prepare input/output names
        input_names = ['inputs_embeds', 'attention_mask']
        for i in range(num_hidden_layers * 2):
            layer_idx = i // 2
            kv_type = "key" if i % 2 == 0 else "value"
            input_names.append(f'past_key_values.{layer_idx}.{kv_type}')

        output_names = ['logits']
        for i in range(num_hidden_layers * 2):
            layer_idx = i // 2
            kv_type = "key" if i % 2 == 0 else "value"
            output_names.append(f'present.{layer_idx}.{kv_type}')

        # Dynamic axes
        dynamic_axes = {
            'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'total_sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }

        for i in range(num_hidden_layers * 2):
            layer_idx = i // 2
            kv_type = "key" if i % 2 == 0 else "value"

            past_name = f'past_key_values.{layer_idx}.{kv_type}'
            dynamic_axes[past_name] = {0: 'batch_size', 2: 'past_sequence_length'}

            present_name = f'present.{layer_idx}.{kv_type}'
            dynamic_axes[present_name] = {0: 'batch_size', 2: 'total_sequence_length'}

        # Export to ONNX
        print(f"\nExporting decoder model to {output_path}...")
        print(f"  (This may take several minutes due to model size... )")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                torch.onnx.export(
                    wrapper,
                    (dummy_inputs_embeds, dummy_attention_mask, *dummy_past_kvs),
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True,
                    verbose=False
                )
            print(f"✓ Decoder model exported successfully")

            # 修正 ONNX 模型中的动态维度
            print("\n  Fixing dynamic dimensions in ONNX model...")
            try:
                import onnx
                from onnx import TensorProto

                model = onnx.load(output_path)

                # 修改输入的形状，将固定维度改为动态维度
                for inp in model.graph.input:
                    if inp.name in dynamic_axes:
                        tensor_type = inp.type.tensor_type
                        dynamic_dims = dynamic_axes[inp.name]

                        # 获取当前维度数量
                        num_dims = len(tensor_type.shape.dim)

                        # 清空并重建 shape
                        del tensor_type.shape.dim[:]

                        # 重新添加维度
                        for dim_idx in range(num_dims):
                            new_dim = tensor_type.shape.dim.add()
                            if dim_idx in dynamic_dims:
                                # 动态维度
                                new_dim.dim_param = dynamic_dims[dim_idx]
                            else:
                                # 固定维度 - 需要获取原始值
                                if inp.name.startswith('past_key_values'):
                                    # past_key_values:  [batch, num_kv_heads, seq_len, head_dim]
                                    if dim_idx == 0:
                                        new_dim.dim_param = 'batch_size'
                                    elif dim_idx == 1:
                                        new_dim.dim_value = num_key_value_heads
                                    elif dim_idx == 2:
                                        new_dim.dim_param = 'past_sequence_length'
                                    elif dim_idx == 3:
                                        new_dim.dim_value = head_dim
                                elif inp.name == 'inputs_embeds':
                                    # [batch, seq_len, hidden_size]
                                    if dim_idx == 2:
                                        new_dim.dim_value = hidden_size
                                elif inp.name == 'attention_mask':
                                    # [batch, total_seq_len]
                                    pass  # 都是动态的

                # 修改输出的形状
                for out in model.graph.output:
                    if out.name in dynamic_axes:
                        tensor_type = out.type.tensor_type
                        dynamic_dims = dynamic_axes[out.name]

                        num_dims = len(tensor_type.shape.dim)
                        del tensor_type.shape.dim[:]

                        for dim_idx in range(num_dims):
                            new_dim = tensor_type.shape.dim.add()
                            if dim_idx in dynamic_dims:
                                new_dim.dim_param = dynamic_dims[dim_idx]
                            else:
                                # 固定维度
                                if out.name.startswith('present'):
                                    # present: [batch, num_kv_heads, total_seq_len, head_dim]
                                    if dim_idx == 0:
                                        new_dim.dim_param = 'batch_size'
                                    elif dim_idx == 1:
                                        new_dim.dim_value = num_key_value_heads
                                    elif dim_idx == 2:
                                        new_dim.dim_param = 'total_sequence_length'
                                    elif dim_idx == 3:
                                        new_dim.dim_value = head_dim
                                elif out.name == 'logits':
                                    # [batch, seq_len, vocab_size]
                                    if dim_idx == 2:
                                        new_dim.dim_value = self.config.text_config.vocab_size

                # 保存修改后的模型
                onnx.save(model, output_path)
                print("  ✓ Dynamic dimensions fixed!")

                # 验证修改
                print("\n  Verifying fixed model:")
                model = onnx.load(output_path)

                # 检查几个示例输入
                print("  Sample input shapes:")
                for inp in model.graph.input[: 2]:
                    shape_info = []
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape_info.append(f"-1({dim.dim_param})")
                        else:
                            shape_info.append(str(dim.dim_value))
                    print(f"    {inp.name}: [{', '.join(shape_info)}]")

                # 检查 past_key_values
                for inp in model.graph.input:
                    if inp.name == "past_key_values. 0.key":
                        shape_info = []
                        for dim in inp.type.tensor_type.shape.dim:
                            if dim.dim_param:
                                shape_info.append(f"-1({dim.dim_param})")
                            else:
                                shape_info.append(str(dim.dim_value))
                        print(f"    {inp.name}: [{', '.join(shape_info)}]")
                        break

                print(f"\n  ✓ All {len(model.graph.input)} inputs verified!")

            except ImportError:
                print("  ⚠️ Warning: onnx package not available.  Dynamic dimensions may not be set correctly.")
                print("  Install with: pip install onnx")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to fix dynamic dimensions: {e}")
                import traceback
                traceback.print_exc()
                raise

        except Exception as e:
            print(f"✗ Export failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def export_all(self, output_dir="./onnx"):
        """Export all components in granite-docling style."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Converting CodeFormulaV2 to ONNX")
        print(f"Output directory: {output_dir}")
        print(f"Following granite-docling-258M-ONNX pattern")
        print(f"{'='*60}")

        # Export vision encoder
        try:
            self.export_vision_encoder(str(output_dir / "vision_encoder.onnx"))
        except Exception as e:
            print(f"✗ Failed to export vision encoder: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Export embed tokens
        try:
            self.export_embed_tokens(str(output_dir / "embed_tokens.onnx"))
        except Exception as e:
            print(f"✗ Failed to export embed tokens: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Export decoder model
        try:
            self.export_decoder_model_merged(str(output_dir / "decoder_model_merged.onnx"))
        except Exception as e:
            print(f"✗ Failed to export decoder model: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Save config and processor
        try:
            print("\n=== Saving Config and Processor ===")
            self.config.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)
            print("✓ Config and processor saved")
        except Exception as e:
            print(f"✗ Failed to save config/processor: {e}")
            raise

        print(f"\n{'='*60}")
        print(f"Export completed!")
        print(f"Files in {output_dir}:")
        for file in sorted(output_dir.glob("*")):
            print(f"  - {file.name}")
        print(f"{'='*60}")


def verify_onnx_models(onnx_dir):
    """Verify the exported ONNX models."""
    try:
        import onnx
        import onnxruntime as ort

        print(f"\n{'='*60}")
        print("Verifying ONNX Models")
        print(f"{'='*60}")

        onnx_dir = Path(onnx_dir)

        for onnx_file in sorted(onnx_dir.glob("*.onnx")):
            print(f"\n{onnx_file.name}:")

            # Load and check
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            print("  ✓ Model is valid")

            # Load with ONNX Runtime
            session = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
            print("  ✓ ONNX Runtime can load")

            # Show inputs/outputs
            print("  Inputs:")
            for inp in session.get_inputs()[: 5]:  # Show first 5 to avoid clutter
                print(f"    - {inp.name}: {inp.shape}")
            if len(session.get_inputs()) > 5:
                print(f"    ... and {len(session.get_inputs()) - 5} more")

            print("  Outputs:")
            for out in session.get_outputs()[:5]:  # Show first 5
                print(f"    - {out.name}: {out.shape}")
            if len(session. get_outputs()) > 5:
                print(f"    ...  and {len(session.get_outputs()) - 5} more")

        return True
    except ImportError:
        print("\n⚠ onnx or onnxruntime not installed.  Skipping verification.")
        print("Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

a_model_id = r"D:\models\CodeFormulaV2"
a_output_dir=r"D:\models\onnx\CodeFormulaV2"

def main():
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = a_model_id
        print(f"Using model path:  {model_path}")
        converter = CodeFormulaV2ONNXConverter(model_id=model_path)
    else:
        converter = CodeFormulaV2ONNXConverter(model_id=a_model_id)

    # Load and export
    converter.load_model()
    converter.export_all(output_dir=a_output_dir)

    # Verify
    verify_onnx_models(a_output_dir)


if __name__ == "__main__":
    main()



'''
# 2026年1月21日 21:27:20
pip list
Package                  Version
------------------------ -----------
absl-py                  2.3.1
addict                   2.4.0
aiohappyeyeballs         2.6.1
aiohttp                  3.12.14
aiosignal                1.4.0
aistudio-sdk             0.3.8
annotated-types          0.7.0
antlr4-python3-runtime   4.9.3
anyio                    4.9.0
astor                    0.8.1
asttokens                3.0.0
attrs                    25.3.0
audioread                3.0.1
av                       15.0.0
babel                    2.17.0
bce-python-sdk           0.9.42
blinker                  1.9.0
blobfile                 3.0.0
bokeh                    3.7.3
boltons                  25.0.0
cachetools               6.1.0
certifi                  2025.4.26
cffi                     1.17.1
chardet                  5.2.0
charset-normalizer       3.4.2
click                    8.2.1
colorama                 0.4.6
colored                  2.3.0
coloredlogs              15.0.1
colorlog                 6.9.0
contourpy                1.3.3
cssselect                1.3.0
cssutils                 2.11.1
cycler                   0.12.1
datasets                 3.6.0
decorator                5.2.1
dill                     0.3.4
docling                  2.69.0
einops                   0.8.1
et_xmlfile               2.0.0
executing                2.2.0
fastapi                  0.116.1
filelock                 3.18.0
Flask                    3.1.1
flask-babel              4.0.0
flatbuffers              25.2.10
fonttools                4.59.0
frozenlist               1.7.0
fsspec                   2025.3.0
ftfy                     6.3.1
future                   1.0.0
GPUtil                   1.4.0
h11                      0.16.0
hf_transfer              0.1.9
httpcore                 1.0.9
httpx                    0.28.1
huggingface-hub          0.34.4
humanfriendly            10.0
idna                     3.10
imagesize                1.4.1
intervaltree             3.1.0
ipython                  9.4.0
ipython_pygments_lexers  1.1.1
itsdangerous             2.2.0
jedi                     0.19.2
jieba                    0.42.1
Jinja2                   3.1.6
joblib                   1.5.1
kiwisolver               1.4.8
lazy_loader              0.4
librosa                  0.11.0
lightning-utilities      0.15.0
ligo-segments            1.4.0
llvmlite                 0.44.0
lxml                     6.0.0
markdown-it-py           3.0.0
MarkupSafe               3.0.2
matplotlib               3.10.3
matplotlib-inline        0.1.7
mdurl                    0.1.2
mido                     1.3.3
ml_dtypes                0.5.3
modelscope               1.28.1
more-itertools           10.7.0
mpmath                   1.3.0
msgpack                  1.1.1
multidict                6.6.3
multiprocess             0.70.12.2
narwhals                 2.0.1
networkx                 3.5
note-seq                 0.0.5
numba                    0.61.2
numpy                    1.26.4
nvidia-cublas-cu12       12.6.4.1
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12        9.5.1.17
nvidia-cufft-cu12        11.3.0.4
nvidia-curand-cu12       10.3.7.77
nvidia-cusolver-cu12     11.7.1.2
nvidia-cusparse-cu12     12.5.4.2
nvidia-nvjitlink-cu12    12.8.93
omegaconf                2.3.0
onnx                     1.17.0
onnx_graphsurgeon        0.5.8
onnxoptimizer            0.3.13
onnxruntime              1.22.1
onnxruntime-gpu          1.22.0
opencv-contrib-python    4.10.0.84
opencv-python            4.11.0.86
openpyxl                 3.1.5
opt-einsum               3.3.0
optimum                  2.0.0
optimum-onnx             0.0.3
packaging                25.0
paddle2onnx              2.0.2rc3
paddlefsl                1.1.0
paddlenlp                3.0.0b4
paddlepaddle-gpu         3.2.0
paddlesde                0.2.5
paddlex                  3.3.3
pandas                   2.3.0
parameterized            0.9.0
parso                    0.8.4
pillow                   11.3.0
pip                      25.1
platformdirs             4.3.8
polygraphy               0.49.24
pooch                    1.8.2
ppdiffusers              0.29.0
premailer                3.10.0
pretty_midi              0.2.10
prettytable              3.16.0
prompt_toolkit           3.0.51
propcache                0.3.2
protobuf                 6.31.1
psutil                   7.0.0
pure_eval                0.2.3
py-cpuinfo               9.0.0
pyarrow                  21.0.0
pyclipper                1.3.0.post6
pycparser                2.22
pycryptodome             3.23.0
pycryptodomex            3.23.0
pydantic                 2.11.7
pydantic_core            2.33.2
pydub                    0.25.1
Pygments                 2.19.2
pyparsing                3.2.3
pypdfium2                4.30.1
pyreadline3              3.5.4
python-bidi              0.6.7
python-dateutil          2.9.0.post0
pytorch-lightning        2.5.2
pytz                     2025.2
PyYAML                   6.0.2
rapidocr-onnxruntime     1.3.25
rarfile                  4.2
regex                    2024.11.6
requests                 2.32.4
requests-mock            1.12.1
rich                     14.1.0
ruamel.yaml              0.18.14
ruamel.yaml.clib         0.2.12
safetensors              0.6.2
scikit-learn             1.7.0
scipy                    1.16.0
seaborn                  0.13.2
sentencepiece            0.2.0
seqeval                  1.2.2
setuptools               78.1.1
shapely                  2.1.1
shellingham              1.5.4
silpa_common             0.3
simplejson               3.20.1
six                      1.17.0
sniffio                  1.3.1
sortedcontainers         2.4.0
soundex                  1.1.3
soundfile                0.13.1
soxr                     0.5.0.post1
stack-data               0.6.3
starlette                0.47.2
sympy                    1.14.0
threadpoolctl            3.6.0
tiktoken                 0.9.0
tokenizers               0.21.4
torch                    2.7.1
torchmetrics             1.8.0
torchvision              0.22.1
tornado                  6.5.1
tqdm                     4.67.1
traitlets                5.14.3
trampoline               0.1.2
transformers             4.51.3
typer                    0.16.0
typing_extensions        4.14.0
typing-inspection        0.4.1
tzdata                   2025.2
ujson                    5.10.0
ultralytics              8.3.170
ultralytics-thop         2.0.14
Unidecode                1.4.0
urllib3                  1.26.20
uvicorn                  0.35.0
visualdl                 2.5.3
wcwidth                  0.2.13
Werkzeug                 3.1.3
wheel                    0.45.1
xxhash                   3.5.0
xyzservices              2025.4.0
yarl                     1.20.1
'''