package io.github.flux.llava;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Autoregressive decoder for LLaVA-OneVision-Qwen2 with KV-cache.
 * Uses decoder_model_merged.onnx.
 *
 * Inputs:
 *   inputs_embeds [batch, seq, hidden_size]
 *   attention_mask [batch, total_seq]
 *   position_ids [batch, seq]  (2D RoPE, not 3D like Qwen3-VL)
 *   past_key_values.{0-L}.{key,value} [batch, num_kv_heads, past_seq, head_dim]
 *
 * Outputs:
 *   logits [batch, seq, vocab_size]
 *   present.{0-L}.{key,value} [batch, num_kv_heads, total_seq, head_dim]
 *
 * Architecture constants from config.json:
 *   num_layers: 24, num_kv_heads: 2, head_dim: 64, hidden_size: 896
 */
public class LlavaOneVisionDecoderModel implements AutoCloseable {

    private final int numLayers;
    private final int numKvHeads;
    private final int headDim;
    private final int hiddenSize;

    private static final long STOP_TOKEN_IM_END = 151645L;
    private static final long STOP_TOKEN_ENDOFTEXT = 151643L;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Map<String, OnnxTensor> outputNames;
    private final int maxNewTokens;

    public LlavaOneVisionDecoderModel(final String modelFile,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final int maxNewTokens,
                                      final int numLayers,
                                      final int numKvHeads,
                                      final int headDim,
                                      final int hiddenSize) {
        this.maxNewTokens = maxNewTokens;
        this.numLayers = numLayers;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.hiddenSize = hiddenSize;
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            // Build output name map for efficient access
            this.outputNames = new HashMap<>();
            for (String name : session.getOutputNames()) {
                outputNames.put(name, null);
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Autoregressive decoding with KV-cache.
     *
     * @param inputsEmbeds [batch, seq, hidden_size] with vision features already merged
     * @param attentionMask [batch, seq]
     * @param positionIds [batch, seq] 2D position IDs (standard RoPE)
     * @param embedModel for embedding subsequent tokens
     * @return generated token IDs for each batch element (excluding EOS)
     */
    public long[][] predict(float[][][] inputsEmbeds,
                            long[][] attentionMask,
                            long[][] positionIds,
                            LlavaOneVisionEmbedModel embedModel) throws OrtException {
        int batchSize = inputsEmbeds.length;

        // --- Prefill ---
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(inputsEmbeds, env));
        prefillInputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));
        prefillInputs.put("position_ids", ArrayUtil.createOnnxTensor(positionIds, env));

        // Empty KV cache for prefill
        for (int i = 0; i < numLayers; i++) {
            OnnxTensor emptyK = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batchSize, numKvHeads, 0, headDim});
            OnnxTensor emptyV = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batchSize, numKvHeads, 0, headDim});
            prefillInputs.put(String.format(Locale.ROOT, "past_key_values.%d.key", i), emptyK);
            prefillInputs.put(String.format(Locale.ROOT, "past_key_values.%d.value", i), emptyV);
        }

        OrtSession.Result prefillResult = session.run(prefillInputs);

        float[][][] logits = (float[][][]) prefillResult.get(0).getValue();
        float[][][][][] kvCache = new float[numLayers * 2][][][][];
        for (int i = 0; i < numLayers; i++) {
            kvCache[2 * i] = (float[][][][])(Object) prefillResult
                    .get(String.format(Locale.ROOT, "present.%d.key", i)).get().getValue();
            kvCache[2 * i + 1] = (float[][][][])(Object) prefillResult
                    .get(String.format(Locale.ROOT, "present.%d.value", i)).get().getValue();
        }
        prefillResult.close();
        OnnxUtil.closeTensors(prefillInputs);

        // First token from prefill
        long[] firstTokens = new long[batchSize];
        for (int b = 0; b < batchSize; b++) {
            firstTokens[b] = ArrayUtil.argmax(logits[b][logits[b].length - 1]);
        }

        long[][] generatedTokens = new long[batchSize][];
        for (int b = 0; b < batchSize; b++) {
            generatedTokens[b] = new long[]{firstTokens[b]};
        }

        long[][] nextTokenIds = new long[batchSize][1];
        for (int b = 0; b < batchSize; b++) {
            nextTokenIds[b][0] = firstTokens[b];
        }

        boolean[] finished = new boolean[batchSize];
        for (int b = 0; b < batchSize; b++) {
            if (isStopToken(firstTokens[b])) {
                finished[b] = true;
            }
        }

        // Track current total sequence length and max position
        int totalLen = attentionMask[0].length;
        long maxPosition = max2D(positionIds);

        // --- Decode loop ---
        for (int step = 0; step < maxNewTokens - 1; step++) {
            if (ArrayUtil.allTrue(finished)) break;

            totalLen += 1;
            maxPosition += 1;

            float[][][] nextEmbeds = embedModel.predict(nextTokenIds);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(nextEmbeds, env));
            inputs.put("attention_mask", ArrayUtil.createOnnxTensor(ArrayUtil.ones(batchSize, totalLen), env));

            // Position IDs for decode: all batches share the same position (text token)
            long[][] decodePosIds = new long[batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                decodePosIds[b][0] = maxPosition;
            }
            inputs.put("position_ids", ArrayUtil.createOnnxTensor(decodePosIds, env));

            for (int i = 0; i < numLayers; i++) {
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.key", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i], env));
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.value", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i + 1], env));
            }

            OrtSession.Result stepResult = session.run(inputs);

            logits = (float[][][]) stepResult.get(0).getValue();
            for (int i = 0; i < numLayers; i++) {
                kvCache[2 * i] = (float[][][][])(Object) stepResult
                        .get(String.format(Locale.ROOT, "present.%d.key", i)).get().getValue();
                kvCache[2 * i + 1] = (float[][][][])(Object) stepResult
                        .get(String.format(Locale.ROOT, "present.%d.value", i)).get().getValue();
            }
            stepResult.close();
            OnnxUtil.closeTensors(inputs);

            long[][] nextIds = new long[batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                if (finished[b]) continue;
                long nextToken = ArrayUtil.argmax(logits[b][0]);
                nextIds[b][0] = nextToken;
                generatedTokens[b] = ArrayUtil.concat(generatedTokens[b], new long[]{nextToken});
                if (isStopToken(nextToken)) {
                    finished[b] = true;
                }
            }
            nextTokenIds = nextIds;
        }

        return generatedTokens;
    }

    private boolean isStopToken(long tokenId) {
        return tokenId == STOP_TOKEN_IM_END || tokenId == STOP_TOKEN_ENDOFTEXT;
    }

    private long max2D(long[][] arr) {
        long max = Long.MIN_VALUE;
        for (long[] d1 : arr) {
            for (long v : d1) {
                if (v > max) max = v;
            }
        }
        return max;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
