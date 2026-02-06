package io.github.flux.lightonocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
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
 * LightOnOCR-2-1B model orchestrator.
 * Pipeline: image → vision encoder → embed tokens → merge image features → decoder (KV-cache) → text
 *
 * Architecture: Pixtral ViT (24 layers) + PatchMerger + Qwen3 decoder (28 layers).
 * Image tokens arranged as grid of <|image_pad|> with <|vision_pad|> breaks and <|vision_end|>.
 */
public class LightOnOcrModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "LightOnOCR-2-1B",
            "LightOnOCR-2-1B-ONNX"
    );

    private static final long IMAGE_TOKEN_ID = 151655L;  // <|image_pad|>

    private final LightOnOcrEncoderModel encoderModel;
    private final LightOnOcrEmbedModel embedModel;
    private final LightOnOcrDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    public LightOnOcrModel(final String modelRootDir,
                           final String modelName,
                           final int gpuIndex,
                           final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new LightOnOcrEncoderModel(
                    new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.embedModel = new LightOnOcrEmbedModel(
                    new File(modelDir, "embed_tokens.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new LightOnOcrDecoderModel(
                    new File(modelDir, "decoder_model_merged.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> pprs,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        try {
            // Get image dimensions from the first PreProcessResult's NDArray
            // Each NDArray is [3, H, W] from the image processor
            List<NDArray> pixelValuesList = PreProcessResult.getNDArrays(pprs);

            // Compute the size config from the processed image dimensions
            // NDArray shape is [3, H, W]
            long[] shape = pixelValuesList.get(0).getShape().getShape();
            int processedH = (int) shape[1];
            int processedW = (int) shape[2];

            // Token grid (floor division by effective patch size 28)
            int effectivePatchSize = 28;
            int numRows = processedH / effectivePatchSize;
            int numCols = processedW / effectivePatchSize;

            // Build flat image tokens (no vision_pad/vision_end breaks)
            int numImageTokens = numRows * numCols;
            String imageTokens = "<|image_pad|>".repeat(numImageTokens);

            // Build full prompt
            String prompt = "<|im_start|>system<|im_end|>\n"
                    + "<|im_start|>user\n"
                    + imageTokens
                    + "<|im_end|>\n"
                    + "<|im_start|>assistant\n";

            // Tokenize
            long[] oneInputIds = tokenizer.encode(prompt).getIds();
            long[][] inputIds = new long[pprs.size()][];
            for (int i = 0; i < pprs.size(); i++) {
                inputIds[i] = oneInputIds;
            }

            // Vision encoder
            float[][][] imageFeatures = encoderModel.predict(pixelValuesList);

            // Token embedding
            float[][][] inputsEmbeds = embedModel.predict(inputIds);

            // Merge image features into text embeddings
            mergeImageFeatures(inputIds, inputsEmbeds, imageFeatures);

            // Attention mask (all 1s)
            int batchSize = pprs.size();
            int seqLen = oneInputIds.length;
            long[][] attentionMask = new long[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    attentionMask[b][s] = 1L;
                }
            }

            // Autoregressive decoding
            long[][] genIds = decoderModel.predict(
                    imageFeatures, inputIds, inputsEmbeds,
                    attentionMask, embedModel);

            // Decode tokens to text
            List<TextResult> results = new ArrayList<>();
            for (long[] tokens : genIds) {
                String text = tokenizer.decode(tokens);
                results.add(new TextResult(text, tokens, -1));
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Replace embeddings at image_pad token positions with vision encoder features.
     */
    private void mergeImageFeatures(long[][] inputIds,
                                    float[][][] inputsEmbeds,
                                    float[][][] imageFeatures) {
        int batchSize = inputIds.length;
        int seqLen = inputIds[0].length;

        for (int b = 0; b < batchSize; b++) {
            int featureIdx = 0;
            for (int pos = 0; pos < seqLen; pos++) {
                if (inputIds[b][pos] == IMAGE_TOKEN_ID) {
                    inputsEmbeds[b][pos] = imageFeatures[b][featureIdx];
                    featureIdx++;
                }
            }
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(null, LightOnOcrImageProcessor.process(rgbMat, matManager, ndManager));
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}

