// this code is convert from https://github.com/huggingface/transformers
// transformers's source code IS Licensed under the Apache License Version 2.0
/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.formula.granite;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.TextResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import io.github.flux.model.FormulaRecognitionModel;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class GraniteDoclingFormulaModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "CodeFormulaV2",
            "granite-docling-258M"
    );

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, GraniteDoclingFormulaModel::new);
    }

    private final GraniteDoclingEncoderModel encoderModel;
    private final GraniteDoclingEmbedModel embedModel;
    private final GraniteDoclingDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final int maxLength;
    private final long eos_token_id;
    private final String query;
    private final String chat_template;

    public GraniteDoclingFormulaModel(final String modelRootDir,
                                      final String modelName,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final int maxLength,
                                      final Map<String, Object> customParams) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }

        this.maxLength = maxLength;
        if ("CodeFormulaV2".equals(modelName)) {
            chat_template = "<|start_of_role|>user<|end_of_role|><image>%s<|end_of_text|>\n<|start_of_role|>assistant:";
        } else {
            chat_template = "<|start_of_role|>user<|end_of_role|><image>%s<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        }
        if ("CodeFormulaV2".equals(modelName)) {
            this.eos_token_id = 100338;
            this.query = "<formula>";
        } else {
            this.eos_token_id = 100257;
            this.query = "Convert formula to LaTeX.";
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new GraniteDoclingEncoderModel(new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.embedModel = new GraniteDoclingEmbedModel(new File(modelDir, "embed_tokens.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new GraniteDoclingDecoderModel(
                    new File(
                            modelDir,
                            "decoder_model_merged.onnx"
                    ).getAbsolutePath(),
                    gpuIndex,
                    env,
                    OnnxJavaType.FLOAT
            );
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        List<TextResult> results = new ArrayList<>();
        for (PreProcessResult ppr : mats) {
            results.add(_predict(chat_template, matManager, ppr.mat(), manager));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(rgbMat, null);
    }

    private TextResult _predict(String chat_template, MatManager matManager, Mat image, NDManager manager) {
        try {
            long image_token_id = 100270;
            int num_hidden_layers = 30;
            int num_key_value_heads = 3;
            int head_dim = 64;

            Map<String, NDArray> past_key_values = new HashMap<>();
            for (String kv : new String[]{"key", "value"}) {
                for (int layer = 0; layer < num_hidden_layers; layer++) {
                    past_key_values.put(String.format(Locale.ROOT,
                                    "past_key_values.%d.%s", layer, kv),
                            manager.zeros(new Shape(1, num_key_value_heads, 0, head_dim), DataType.FLOAT32));
                }
            }

            NDArray imgNdArray = ImageUtil.toNDArrayUint8(image, manager);
            String requestText = Idefics3Processor.apply_chat_template(chat_template, query);
            // cost ~40ms
            Idefics3PreProcessResult preResult = Idefics3ImageProcessor.process(tokenizer, requestText, matManager, imgNdArray, manager);
            long[] input_ids_long = preResult.input_ids();
            NDArray input_ids = manager.create(input_ids_long, new Shape(1, input_ids_long.length));
            long[] attention_mask_long = preResult.attention_mask();

            float[][][][] image_feature_floats = encoderModel.predict(preResult.pixel_values(), preResult.pixel_attention_mask());

            NDArray attention_mask = manager.create(attention_mask_long, new Shape(1, attention_mask_long.length));
            long[] generated_tokens = new long[]{};
            for (int i = 0; i < maxLength; i++) {
                float[][][] inputs_embed_floats = embedModel.predict(input_ids);
                NDArray inputs_embeds = ArrayUtil.toNDArray(manager, inputs_embed_floats);
                if (i == 0) {
                    NDArray image_features = ArrayUtil.toNDArray(manager, image_feature_floats);
                    mergeTextAndVisionEmbeddings(input_ids, image_token_id,
                            inputs_embeds, image_features);
                }

                GraniteDoclingDecodeResult decodeResult = decoderModel.predict(inputs_embeds, attention_mask,
                        past_key_values);
                float[][][] logits_floats = decodeResult.logits();
                Map<String, float[][][][]> present_key_values = decodeResult.present_key_values();

                past_key_values.forEach((_, pkv) -> IOUtil.close(pkv));
                for (int layer = 0; layer < num_hidden_layers; layer++) {
                    for (String kv : new String[]{"key", "value"}) {
                        float[][][][] present_floats = present_key_values.get(String.format(Locale.ROOT,
                                "present.%d.%s", layer, kv));
                        past_key_values.put(String.format(Locale.ROOT,
                                        "past_key_values.%d.%s", layer, kv),
                                ArrayUtil.toNDArray(manager, present_floats));
                    }
                }

                NDArray logits = ArrayUtil.toNDArray(manager, logits_floats);

                NDArray lastLogits = logits.get(new NDIndex(":, -1, :"));
                IOUtil.close(logits);
                NDArray inputIds = lastLogits.argMax(-1);
                IOUtil.close(lastLogits);
                NDArray next_tokens = inputIds.expandDims(1);
                IOUtil.close(inputIds);
                input_ids = next_tokens;
                NDArray ones = manager.ones(new Shape(1, 1), attention_mask.getDataType());
                IOUtil.close(lastLogits);
                attention_mask = NDArrays.concat(new NDList(attention_mask, ones), -1);
                long[] nextTokenIds = next_tokens.toLongArray();
                generated_tokens = ArrayUtil.concat(generated_tokens, nextTokenIds);
                if (input_ids.eq(eos_token_id).all().getBoolean()) {
                    break;
                }
            }

            image.release();
            past_key_values.forEach((_, pkv) -> IOUtil.close(pkv));
            IOUtil.close(imgNdArray);
            IOUtil.close(input_ids);
            IOUtil.close(attention_mask);
            return new TextResult(tokenizer.decode(generated_tokens, false), generated_tokens, -1);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private void mergeTextAndVisionEmbeddings(NDArray inputIds, long imageTokenId,
                                              NDArray inputsEmbeds, NDArray imageFeatures) {

        // 1. 生成 mask: (1, seqLen)
        NDArray mask = inputIds.eq(imageTokenId);  // boolean mask

        // 2. 找到 True 的位置索引
        NDArray positions = mask.nonzero();  // shape: (num_patches, 2)
        NDArray tokenPositions = positions.get(":, 1");  // 取第二列（序列维度）

        // 3. reshape image features 确保形状匹配
        NDArray reshapedImageFeatures = imageFeatures.reshape(-1, inputsEmbeds.getShape().get(2));

        // 4. 替换 embedding
        for (int i = 0; i < tokenPositions.size(); i++) {
            long pos = tokenPositions.getLong(i);
            NDIndex idx = new NDIndex("0," + pos + ",:");
            inputsEmbeds.set(idx, reshapedImageFeatures.get(i));
        }

        IOUtil.close(tokenPositions);
        IOUtil.close(positions);
        IOUtil.close(mask);
        IOUtil.close(inputIds);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }

}
