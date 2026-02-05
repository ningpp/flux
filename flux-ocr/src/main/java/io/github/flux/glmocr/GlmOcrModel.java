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
package io.github.flux.glmocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * GLM-OCR Model for end-to-end document OCR.
 * 
 * Uses a vision-language architecture with:
 * - Vision encoder for image feature extraction
 * - Embedding model for token embeddings
 * - Unified LLM decoder with KV-cache for autoregressive generation
 * 
 * Model files expected in model directory:
 * - vision_encoder.onnx: Vision encoder model
 * - embedding.onnx: Token embedding model
 * - llm_unified.onnx: Unified LLM decoder model (or llm_unified_fp16.onnx for FP16)
 * - tokenizer.json: HuggingFace tokenizer config
 */
public class GlmOcrModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "GLM-OCR"
    );

    private static final int IMAGE_TOKEN_ID = 151329;  // <|begin_of_image|>
    private static final int IMAGE_PAD_TOKEN_ID = 151330;  // <|image_pad|>
    private static final int END_IMAGE_TOKEN_ID = 151331;  // <|end_of_image|>
    private static final int NUM_IMAGE_TOKENS = 256;  // Number of image token placeholders
    
    private final GlmOcrVisionEncoderModel encoderModel;
    private final GlmOcrEmbedModel embedModel;
    private final GlmOcrDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    /**
     * Create GLM-OCR model.
     *
     * @param modelRootDir root directory containing model subdirectories
     * @param modelName model name (e.g., "GLM-OCR")
     * @param gpuIndex GPU index (-1 for CPU)
     * @param env ONNX Runtime environment
     */
    public GlmOcrModel(final String modelRootDir,
                       final String modelName,
                       final int gpuIndex,
                       final OrtEnvironment env) {
        this(modelRootDir, modelName, gpuIndex, env, false);
    }

    /**
     * Create GLM-OCR model with optional FP16 precision.
     *
     * @param modelRootDir root directory containing model subdirectories
     * @param modelName model name (e.g., "GLM-OCR")
     * @param gpuIndex GPU index (-1 for CPU)
     * @param env ONNX Runtime environment
     * @param useFp16 whether to use FP16 model for reduced memory
     */
    public GlmOcrModel(final String modelRootDir,
                       final String modelName,
                       final int gpuIndex,
                       final OrtEnvironment env,
                       final boolean useFp16) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new GlmOcrVisionEncoderModel(
                    new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            
            this.embedModel = new GlmOcrEmbedModel(
                    new File(modelDir, "embedding.onnx").getAbsolutePath(),
                    gpuIndex, env);
            
            String llmModelName = useFp16 ? "llm_unified_fp16.onnx" : "llm_unified.onnx";
            this.decoderModel = new GlmOcrDecoderModel(
                    new File(modelDir, llmModelName).getAbsolutePath(),
                    gpuIndex,
                    env,
                    4096  // max length
            );
            
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> pprs,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        // Build prompt with image placeholders
        String imgPadStr = "<|image_pad|>".repeat(NUM_IMAGE_TOKENS);
        String prompt = "<|system|>\nYou are an OCR assistant. Extract all text from the image.<|end|>\n"
                + "<|user|>\n<|begin_of_image|>" + imgPadStr + "<|end_of_image|>\n"
                + "Please perform OCR on this image and return all text content.<|end|>\n"
                + "<|assistant|>\n";
        
        long[] oneInputIds = tokenizer.encode(prompt).getIds();
        long[][] inputIds = new long[pprs.size()][];
        for (int i = 0; i < pprs.size(); i++) {
            inputIds[i] = oneInputIds.clone();
        }

        try {
            // Encode images
            float[][][] imageFeatures = encoderModel.predict(PreProcessResult.getNDArrays(pprs));
            
            // Get text embeddings
            float[][][] inputsEmbeds = embedModel.predict(inputIds);
            
            // Merge image features into embeddings at image token positions
            prepareInputsEmbeds(inputIds, IMAGE_PAD_TOKEN_ID, imageFeatures, inputsEmbeds);

            // Generate output tokens
            long[][] genIds = decoderModel.predict(
                    imageFeatures,
                    inputIds,
                    inputsEmbeds,
                    embedModel,
                    ndManager
            );

            // Decode tokens to text
            List<TextResult> textResults = new ArrayList<>();
            for (long[] tokens : genIds) {
                String text = tokenizer.decode(tokens);
                textResults.add(new TextResult(text, tokens, -1));
            }
            return textResults;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Merge image features into text embeddings at image pad token positions.
     *
     * @param inputIds input token IDs
     * @param imageTokenIndex the token ID for image padding
     * @param imageFeatures vision encoder outputs [batch, num_patches, hidden]
     * @param inputsEmbeds text embeddings to be modified in-place [batch, seq_len, hidden]
     */
    private void prepareInputsEmbeds(long[][] inputIds,
                                     long imageTokenIndex,
                                     float[][][] imageFeatures,
                                     float[][][] inputsEmbeds) {
        int batchSize = inputIds.length;
        int seqLen = inputIds[0].length;

        for (int i = 0; i < batchSize; i++) {
            int imageFeatureIdx = 0;
            for (int pos = 0; pos < seqLen; pos++) {
                if (inputIds[i][pos] == imageTokenIndex) {
                    // Replace text embedding with image feature
                    if (imageFeatureIdx < imageFeatures[i].length) {
                        inputsEmbeds[i][pos] = imageFeatures[i][imageFeatureIdx];
                        imageFeatureIdx++;
                    }
                }
            }
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(null, GlmOcrImageProcessor.process(rgbMat, matManager, ndManager));
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}
