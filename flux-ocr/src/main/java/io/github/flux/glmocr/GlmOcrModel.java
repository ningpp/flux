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
import ai.onnxruntime.OrtException;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * GLM-OCR Model for end-to-end document OCR.
 * 
 * Uses a vision-language architecture with:
 * - Vision encoder for image feature extraction (unique patch-based input)
 * - Embedding model for token embeddings
 * - Unified LLM decoder with KV-cache for autoregressive generation
 * 
 * Model files expected in model directory:
 * - vision_encoder.onnx: Vision encoder model
 * - embedding.onnx: Token embedding model
 * - llm_unified.onnx: Unified LLM decoder model (or llm_unified_fp16.onnx for FP16)
 * - tokenizer.json: HuggingFace tokenizer config
 */
public class GlmOcrModel implements AutoCloseable {

    public static final Set<String> MODEL_NAMES = Set.of(
            "GLM-OCR"
    );

    // Token IDs for GLM-OCR
    private static final long IMAGE_TOKEN_ID = 59280L;       // <|image|>
    private static final long BEGIN_IMAGE_TOKEN_ID = 59256L; // <|begin_of_image|>
    private static final long END_IMAGE_TOKEN_ID = 59257L;   // <|end_of_image|>
    private static final long EOS_TOKEN_ID = 151643L;        // End of sequence
    
    // Merge size for calculating number of image tokens
    private static final int MERGE_SIZE = 2;
    
    private final GlmOcrVisionEncoderModel encoderModel;
    private final GlmOcrEmbedModel embedModel;
    private final GlmOcrDecoder decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    /**
     * Create GLM-OCR model with optional FP16 precision.
     *
     * @param modelRootDir root directory containing model subdirectories
     * @param modelName model name (e.g., "GLM-OCR")
     * @param gpuIndex GPU index (-1 for CPU)
     * @param env ONNX Runtime environment
     * @param useFp16 whether to use FP16 model for reduced memory (currently not supported)
     */
    public GlmOcrModel(final String modelRootDir,
                       final String modelName,
                       final int gpuIndex,
                       final OrtEnvironment env,
                       final boolean useFp16,
                       final boolean useUnified) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }
        
        if (useFp16) {
            System.err.println("Warning: FP16 mode not yet supported, using FP32 models");
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new GlmOcrVisionEncoderModel(
                    new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            
            this.embedModel = new GlmOcrEmbedModel(
                    new File(modelDir, "embedding.onnx").getAbsolutePath(),
                    gpuIndex, env);

            if (useUnified) {
                this.decoderModel = new GlmOcrDecoderModelUnified(
                        new File(modelDir, "llm_unified.onnx").getAbsolutePath(),
                        gpuIndex,
                        env,
                        1024
                );
            } else {
                // Use separate prefill and decode models for better CUDA compatibility
                this.decoderModel = new GlmOcrDecoderModel(
                        new File(modelDir, "llm_prefill.onnx").getAbsolutePath(),
                        new File(modelDir, "llm_decode.onnx").getAbsolutePath(),
                        gpuIndex,
                        env,
                        1024
                );
            }
            
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir),
                    java.util.Map.of("truncation", "false", "padding", "false"));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Perform OCR on a single image.
     *
     * @param rgbMat input RGB image
     * @param matManager OpenCV Mat resource manager
     * @param ndManager NDArray manager
     * @param prompt OCR prompt (e.g., "OCR:")
     * @return OCR result text
     */
    public TextResult predict(Mat rgbMat, MatManager matManager, NDManager ndManager, String prompt) {
        try {
            // Preprocess image
            GlmOcrImageProcessor.PreprocessResult ppResult = 
                    GlmOcrImageProcessor.process(rgbMat, matManager, ndManager);
            
            // Encode image
            float[][] imageFeatures = encoderModel.predict(ppResult.pixelValues, ppResult.imageGridThw);
            
            // Calculate number of image tokens after merging
            // Python: num_image_tokens = (h_patches / merge_size) * (w_patches / merge_size)
            int hPatches = ppResult.imageGridThw[1];
            int wPatches = ppResult.imageGridThw[2];
            int numImageTokens = (hPatches / MERGE_SIZE) * (wPatches / MERGE_SIZE);
            
            // Build prompt with correct number of image placeholders
            String imgPadStr = "<|image|>".repeat(numImageTokens);
            String fullPrompt = "[gMASK]<sop><|user|>\n"
                    + "<|begin_of_image|>" + imgPadStr + "<|end_of_image|>" + prompt + "<|assistant|>\n";
            
            // Tokenize prompt
            long[] inputIds = tokenizer.encode(fullPrompt).getIds();

            // Get text embeddings
            float[][][] inputsEmbeds = embedModel.predict(new long[][]{inputIds});
            
            // Merge image features into embeddings at image pad token positions
            mergeImageFeatures(inputIds, inputsEmbeds[0], imageFeatures);
            
            // Generate output tokens
            long[][] genIds = decoderModel.predict(
                    expandToHidden(imageFeatures),  // Wrap for batch
                    new long[][]{inputIds},
                    inputsEmbeds,
                    ppResult.imageGridThw,
                    embedModel,
                    ndManager
            );
            
            // Decode tokens to text
            String text = tokenizer.decode(genIds[0]);
            return new TextResult(text, genIds[0], -1);
            
        } catch (OrtException e) {
            throw new FluxException(e);
        }
    }

    /**
     * Perform OCR on a single image with default prompt.
     */
    public TextResult predict(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        return predict(rgbMat, matManager, ndManager, "OCR:");
    }

    /**
     * Perform batch OCR on multiple images.
     * Note: Due to variable sequence lengths, images are processed sequentially.
     *
     * @param images list of RGB images
     * @param matManager OpenCV Mat resource manager
     * @param ndManager NDArray manager
     * @param prompt OCR prompt
     * @return list of OCR results
     */
    public List<TextResult> batchPredict(List<Mat> images, MatManager matManager, NDManager ndManager, String prompt) {
        List<TextResult> results = new ArrayList<>();
        for (Mat image : images) {
            results.add(predict(image, matManager, ndManager, prompt));
        }
        return results;
    }

    /**
     * Merge image features into text embeddings at image pad token positions.
     */
    private void mergeImageFeatures(long[] inputIds, float[][] inputsEmbeds, float[][] imageFeatures) {
        int imageFeatureIdx = 0;
        for (int pos = 0; pos < inputIds.length; pos++) {
            if (inputIds[pos] == IMAGE_TOKEN_ID) {
                if (imageFeatureIdx < imageFeatures.length) {
                    // Copy image features to embedding position
                    System.arraycopy(imageFeatures[imageFeatureIdx], 0, 
                            inputsEmbeds[pos], 0, 
                            Math.min(imageFeatures[imageFeatureIdx].length, inputsEmbeds[pos].length));
                    imageFeatureIdx++;
                }
            }
        }
    }

    /**
     * Expand 2D image features to 3D format expected by decoder.
     */
    private float[][][] expandToHidden(float[][] imageFeatures) {
        // [num_tokens, hidden] -> [1, num_tokens, hidden] for batch=1
        float[][][] result = new float[1][imageFeatures.length][imageFeatures[0].length];
        for (int i = 0; i < imageFeatures.length; i++) {
            System.arraycopy(imageFeatures[i], 0, result[0][i], 0, imageFeatures[i].length);
        }
        return result;
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}
