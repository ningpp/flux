package io.github.flux.qwen3vl;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.qwen3vl.Qwen3VlEncoderModel.EncoderResult;
import io.github.flux.qwen3vl.Qwen3VlImageProcessor.ImageProcessResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Qwen3-VL model orchestrator.
 * Pipeline: image → vision encoder (+ deepstack) → embed tokens → merge image features
 *           → compute MRoPE position_ids → scatter deepstack → decoder (KV-cache) → text
 *
 * Architecture: Qwen3 ViT + DeepStack + Qwen3 decoder.
 * DeepStack count and model constants are determined dynamically from the ONNX models.
 * Image tokens use 3D MRoPE (T/H/W) for position encoding.
 */
public class Qwen3VlModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "Qwen3-VL-2B-Instruct"
    );

    // Special token IDs
    private static final long IM_START = 151644L;
    private static final long IM_END = 151645L;
    private static final long VISION_START = 151652L;
    private static final long VISION_END = 151653L;
    private static final long IMAGE_PAD = 151655L;

    private static final int MERGE_SIZE = 2;

    private final Qwen3VlEncoderModel encoderModel;
    private final Qwen3VlEmbedModel embedModel;
    private final Qwen3VlDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final int hiddenSize;

    /** Stored from preprocessing for use during prediction. */
    private ImageProcessResult lastImageResult;

    public Qwen3VlModel(final String modelRootDir,
                        final String modelName,
                        final int gpuIndex,
                        final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }
        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new Qwen3VlEncoderModel(
                    new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.embedModel = new Qwen3VlEmbedModel(
                    new File(modelDir, "embed_tokens.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));

            // Read model config for dynamic architecture constants
            Type mapType = new TypeToken<Map<String, Object>>() {}.getType();
            Map<String, Object> config = new Gson().fromJson(
                    Files.readString(new File(modelDir, "config.json").toPath(), StandardCharsets.UTF_8),
                    mapType);
            @SuppressWarnings("unchecked")
            Map<String, Object> textConfig = (Map<String, Object>) config.getOrDefault("text_config", Map.of());
            int numLayers = ((Number) textConfig.getOrDefault("num_hidden_layers", 28)).intValue();
            int numKvHeads = ((Number) textConfig.getOrDefault("num_key_value_heads", 8)).intValue();
            int headDim = ((Number) textConfig.getOrDefault("head_dim", 128)).intValue();
            this.hiddenSize = ((Number) textConfig.getOrDefault("hidden_size", 2048)).intValue();
            int numDeepstack = encoderModel.getNumDeepstack();

            this.decoderModel = new Qwen3VlDecoderModel(
                    new File(modelDir, "decoder_model_merged.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096,
                    numLayers, numKvHeads, headDim, this.hiddenSize, numDeepstack);
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
            if (pprs.size() != 1) {
                throw new FluxException("Qwen3-VL currently only supports batch size 1");
            }

            // The ImageProcessResult was stored during processRgb
            ImageProcessResult imageResult = lastImageResult;
            if (imageResult == null) {
                throw new FluxException("Image not preprocessed. Call processRgb first.");
            }

            // 1. Vision encoder
            EncoderResult encoderResult = encoderModel.predict(
                    imageResult.pixelValues(), imageResult.imageGridThw());

            float[][] imageFeatures = encoderResult.imageFeatures();
            int numMergedTokens = imageFeatures.length;

            // 2. Build input_ids
            long[] inputIds = buildInputIds(numMergedTokens);
            int seqLen = inputIds.length;

            // 3. Compute MRoPE position_ids [3, 1, seqLen]
            long[][][] positionIds = computePositionIds(inputIds, imageResult.imageGridThw());

            // 4. Embed tokens
            long[][] inputIds2D = new long[][]{inputIds};
            float[][][] inputsEmbeds = embedModel.predict(inputIds2D);

            // 5. Replace IMAGE_PAD embeddings with vision features
            int featureIdx = 0;
            for (int pos = 0; pos < seqLen; pos++) {
                if (inputIds[pos] == IMAGE_PAD) {
                    System.arraycopy(imageFeatures[featureIdx], 0, inputsEmbeds[0][pos], 0, hiddenSize);
                    featureIdx++;
                }
            }

            // 6. Scatter deepstack features
            float[][][][] deepstackScattered = scatterDeepstack(
                    inputIds, seqLen,
                    encoderResult.deepstackFeatures());

            // 7. Attention mask
            long[][] attentionMask = new long[1][seqLen];
            for (int i = 0; i < seqLen; i++) {
                attentionMask[0][i] = 1L;
            }

            // 8. Decode
            long[][] generatedIds = decoderModel.predict(
                    inputsEmbeds, attentionMask, positionIds,
                    deepstackScattered, embedModel);

            // 9. Decode tokens to text
            List<TextResult> results = new ArrayList<>();
            for (long[] tokens : generatedIds) {
                String text = tokenizer.decode(tokens);
                results.add(new TextResult(text, tokens, -1));
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Build prompt input_ids with the formula recognition template.
     */
    private long[] buildInputIds(int numImageTokens) {
        long[] systemText = tokenizer.encode("system\nYou are a helpful assistant.").getIds();
        long[] userPrefix = tokenizer.encode("user\n").getIds();
        long[] userText = tokenizer.encode("Convert this formula image to LaTeX.\n/no_think").getIds();
        long[] assistantText = tokenizer.encode("assistant\n").getIds();
        long[] newline = tokenizer.encode("\n").getIds();

        List<Long> ids = new ArrayList<>();
        ids.add(IM_START);
        for (long id : systemText) ids.add(id);
        ids.add(IM_END);
        for (long id : newline) ids.add(id);
        ids.add(IM_START);
        for (long id : userPrefix) ids.add(id);
        ids.add(VISION_START);
        for (int i = 0; i < numImageTokens; i++) ids.add(IMAGE_PAD);
        ids.add(VISION_END);
        for (long id : userText) ids.add(id);
        ids.add(IM_END);
        for (long id : newline) ids.add(id);
        ids.add(IM_START);
        for (long id : assistantText) ids.add(id);

        return ids.stream().mapToLong(Long::longValue).toArray();
    }

    /**
     * Compute MRoPE 3D position_ids [3, 1, seqLen].
     * Text tokens: all 3 dims share the same incrementing position.
     * Image tokens: T=constant, H=row, W=col in merged grid.
     */
    private long[][][] computePositionIds(long[] inputIds, long[][] imageGridThw) {
        int seqLen = inputIds.length;
        List<long[][]> segments = new ArrayList<>();
        int st = 0;
        int imageIdx = 0;

        // Find all IMAGE_PAD positions
        for (int i = 0; i < seqLen; i++) {
            if (inputIds[i] == IMAGE_PAD) {
                int imgStart = i;
                // Find the text before this image region
                int textLen = imgStart - st;

                long t = imageGridThw[imageIdx][0];
                long h = imageGridThw[imageIdx][1];
                long w = imageGridThw[imageIdx][2];
                imageIdx++;

                int llmGridT = (int) t;
                int llmGridH = (int) (h / MERGE_SIZE);
                int llmGridW = (int) (w / MERGE_SIZE);

                // Compute st_idx ONCE for this iteration (matching transformers)
                long stIdx = segments.isEmpty() ? 0 : maxOfLastSegment(segments) + 1;

                // Text positions before the IMAGE_PAD region
                if (textLen > 0) {
                    long[][] textPos = new long[3][textLen];
                    for (int d = 0; d < 3; d++) {
                        for (int p = 0; p < textLen; p++) {
                            textPos[d][p] = stIdx + p;
                        }
                    }
                    segments.add(textPos);
                }

                // Vision positions: T/H/W grid
                int numVisTokens = llmGridT * llmGridH * llmGridW;
                long[][] visPos = new long[3][numVisTokens];
                int vIdx = 0;
                for (int gt = 0; gt < llmGridT; gt++) {
                    for (int gh = 0; gh < llmGridH; gh++) {
                        for (int gw = 0; gw < llmGridW; gw++) {
                            visPos[0][vIdx] = gt + textLen + stIdx;
                            visPos[1][vIdx] = gh + textLen + stIdx;
                            visPos[2][vIdx] = gw + textLen + stIdx;
                            vIdx++;
                        }
                    }
                }
                segments.add(visPos);

                // Skip to end of IMAGE_PAD region
                int imgEnd = imgStart;
                while (imgEnd < seqLen && inputIds[imgEnd] == IMAGE_PAD) imgEnd++;
                st = imgEnd;
                i = imgEnd - 1; // loop will increment
            }
        }

        // Remaining text tokens
        if (st < seqLen) {
            long stIdx = segments.isEmpty() ? 0 : maxOfLastSegment(segments) + 1;
            int textLen = seqLen - st;
            long[][] textPos = new long[3][textLen];
            for (int d = 0; d < 3; d++) {
                for (int p = 0; p < textLen; p++) {
                    textPos[d][p] = stIdx + p;
                }
            }
            segments.add(textPos);
        }

        // Concatenate all segments into [3, 1, seqLen]
        long[][][] positionIds = new long[3][1][seqLen];
        int offset = 0;
        for (long[][] seg : segments) {
            int segLen = seg[0].length;
            for (int d = 0; d < 3; d++) {
                System.arraycopy(seg[d], 0, positionIds[d][0], offset, segLen);
            }
            offset += segLen;
        }

        return positionIds;
    }

    private long maxOfLastSegment(List<long[][]> segments) {
        long[][] last = segments.get(segments.size() - 1);
        long max = Long.MIN_VALUE;
        for (long[] dim : last) {
            for (long v : dim) {
                if (v > max) max = v;
            }
        }
        return max;
    }

    /**
     * Scatter deepstack features [num_vis, hidden_size] to [1, seqLen, hidden_size] at IMAGE_PAD positions.
     * Returns [N][1][seqLen][hidden_size] for the N deepstack levels.
     *
     * @param inputIds     input token IDs
     * @param seqLen       sequence length
     * @param deepstackFeatures [N][num_vis][hidden_size] deepstack feature tensors
     */
    private float[][][][] scatterDeepstack(long[] inputIds, int seqLen,
                                           float[][][] deepstackFeatures) {
        int numDs = deepstackFeatures.length;
        float[][][][] result = new float[numDs][1][seqLen][hiddenSize];

        int featureIdx = 0;
        for (int pos = 0; pos < seqLen; pos++) {
            if (inputIds[pos] == IMAGE_PAD) {
                for (int d = 0; d < numDs; d++) {
                    System.arraycopy(deepstackFeatures[d][featureIdx], 0, result[d][0][pos], 0, hiddenSize);
                }
                featureIdx++;
            }
        }
        return result;
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        this.lastImageResult = Qwen3VlImageProcessor.process(rgbMat, matManager);
        // Return a PreProcessResult with null mat/ndarray - we stored the result internally
        return new PreProcessResult(null, null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}
