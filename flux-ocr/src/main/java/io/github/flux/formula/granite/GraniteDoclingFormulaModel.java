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
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
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

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class GraniteDoclingFormulaModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "CodeFormulaV2",
            "granite-docling-258M"
    );

    // Note: This model is registered in FormulaRecognitionModel with special handling for maxLength parameter

    private final GraniteDoclingEncoderModel encoderModel;
    private final GraniteDoclingEmbedModel embedModel;
    private final GraniteDoclingDecoderModel decoderModel;
    private final OrtEnvironment env;
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
                this.env = env;
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
        NDArray imgNdArray = null;
        GraniteDoclingDecodeResult pastDecodeResult = null;
        try {
            long image_token_id = 100270;

            imgNdArray = ImageUtil.toNDArrayUint8(image, manager);
            String requestText = Idefics3Processor.apply_chat_template(chat_template, query);
            Idefics3PreProcessResult preResult = Idefics3ImageProcessor.process(tokenizer, requestText, matManager, imgNdArray, manager);
            long[] input_ids = preResult.input_ids();
            long[] attention_mask = preResult.attention_mask();

            float[][][][] image_feature_floats = encoderModel.predict(preResult.pixel_values(), preResult.pixel_attention_mask());

            List<Long> generated_tokens_list = new ArrayList<>(maxLength);
            for (int i = 0; i < maxLength; i++) {
                long[][] input_ids_batch = new long[][]{input_ids};
                GraniteDoclingDecodeResult decodeResult;
                try (GraniteDoclingEmbedModel.PredictResult embedPredictResult = embedModel.predictTensor(input_ids_batch)) {
                    OnnxTensor inputEmbedsTensor = embedPredictResult.embeddings();
                    if (i == 0) {
                        float[][][] inputEmbedsFloats = (float[][][]) inputEmbedsTensor.getValue();
                        mergeTextAndVisionEmbeddings(input_ids, image_token_id, inputEmbedsFloats, image_feature_floats);
                        try (OnnxTensor mergedInputEmbedsTensor = ArrayUtil.createOnnxTensor(inputEmbedsFloats, env)) {
                            decodeResult = decoderModel.predict(
                                    mergedInputEmbedsTensor,
                                    attention_mask,
                                    pastDecodeResult == null ? null : pastDecodeResult.present_key_values()
                            );
                        }
                    } else {
                        decodeResult = decoderModel.predict(
                                inputEmbedsTensor,
                                attention_mask,
                                pastDecodeResult == null ? null : pastDecodeResult.present_key_values()
                        );
                    }
                }
                float[][][] logits_floats = decodeResult.logits();

                if (pastDecodeResult != null) {
                    pastDecodeResult.close();
                }
                pastDecodeResult = decodeResult;

                long nextTokenId = argmaxLastToken(logits_floats);
                input_ids = new long[]{nextTokenId};
                attention_mask = Arrays.copyOf(attention_mask, attention_mask.length + 1);
                attention_mask[attention_mask.length - 1] = 1L;
                generated_tokens_list.add(nextTokenId);
                if (nextTokenId == eos_token_id) {
                    break;
                }
            }

            matManager.release(image);
            image = null; // mark as released
            long[] generated_tokens = new long[generated_tokens_list.size()];
            for (int i = 0; i < generated_tokens_list.size(); i++) {
                generated_tokens[i] = generated_tokens_list.get(i);
            }
            TextResult result = new TextResult(tokenizer.decode(generated_tokens, false), generated_tokens, -1);
            // Transfer ownership - pastDecodeResult and imgNdArray will be cleaned up by finally
            pastDecodeResult = null;
            imgNdArray = null;
            return result;
        } catch (Exception e) {
            throw new FluxException(e);
        } finally {
            IOUtil.close(imgNdArray);
            IOUtil.close(pastDecodeResult);
            if (image != null) {
                matManager.release(image);
            }
        }
    }

    private long argmaxLastToken(float[][][] logits) {
        float[] values = logits[0][logits[0].length - 1];
        int maxIndex = 0;
        float maxValue = values[0];
        for (int i = 1; i < values.length; i++) {
            float current = values[i];
            if (current > maxValue) {
                maxValue = current;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void mergeTextAndVisionEmbeddings(long[] inputIds, long imageTokenId,
                                              float[][][] inputsEmbeds, float[][][][] imageFeatures) {
        if (imageFeatures.length == 0) {
            return;
        }

        int hiddenSize = inputsEmbeds[0][0].length;
        float[][] flatImageFeatures = flattenImageFeatures(imageFeatures, hiddenSize);

        int imageFeatureIndex = 0;
        for (int tokenIndex = 0; tokenIndex < inputIds.length; tokenIndex++) {
            if (inputIds[tokenIndex] == imageTokenId && imageFeatureIndex < flatImageFeatures.length) {
                System.arraycopy(flatImageFeatures[imageFeatureIndex], 0,
                        inputsEmbeds[0][tokenIndex], 0, inputsEmbeds[0][tokenIndex].length);
                imageFeatureIndex++;
            }
        }
    }

    private float[][] flattenImageFeatures(float[][][][] imageFeatures, int hiddenSize) {
        int totalRows = 0;
        for (float[][][] batch : imageFeatures) {
            for (float[][] group : batch) {
                totalRows += group.length;
            }
        }

        float[][] flat = new float[totalRows][hiddenSize];
        int row = 0;
        for (float[][][] batch : imageFeatures) {
            for (float[][] group : batch) {
                for (float[] vector : group) {
                    int copyLength = Math.min(hiddenSize, vector.length);
                    System.arraycopy(vector, 0, flat[row], 0, copyLength);
                    row++;
                }
            }
        }
        return flat;
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }

}
