package io.github.flux.qwen3vl;

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
import java.util.Set;

/**
 * Autoregressive decoder for Qwen3-VL with KV-cache and pre-scattered DeepStack.
 * Uses decoder_model_merged.onnx.
 *
 * Inputs:
 *   inputs_embeds [batch, seq, hidden_size]
 *   attention_mask [batch, total_seq]
 *   position_ids [3, batch, seq]  (MRoPE: T/H/W)
 *   deepstack_scattered_0..N-1 [batch, seq, hidden_size]
 *   past_key_values.{0-L}.{key,value} [batch, num_kv_heads, past_seq, head_dim]
 *
 * Outputs:
 *   logits [batch, seq, vocab_size]
 *   present.{0-L}.{key,value} [batch, num_kv_heads, total_seq, head_dim]
 *
 * Architecture constants and deepstack count are configurable.
 */
public class Qwen3VlDecoderModel implements AutoCloseable {

    private final int numLayers;
    private final int numKvHeads;
    private final int headDim;
    private final int hiddenSize;
    private final int numDeepstack;

    private static final long STOP_TOKEN_IM_END = 151645L;
    private static final long STOP_TOKEN_ENDOFTEXT = 151643L;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;
    private final int maxNewTokens;

    public Qwen3VlDecoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               final int maxNewTokens,
                               final int numLayers,
                               final int numKvHeads,
                               final int headDim,
                               final int hiddenSize,
                               final int numDeepstack) {
        this.maxNewTokens = maxNewTokens;
        this.numLayers = numLayers;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.hiddenSize = hiddenSize;
        this.numDeepstack = numDeepstack;
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Autoregressive decoding with KV-cache.
     *
     * @param inputsEmbeds    [batch, seq, hidden_size] with vision features already merged
     * @param attentionMask   [batch, seq]
     * @param positionIds     [3, batch, seq] MRoPE position IDs
     * @param deepstackScattered N tensors each [1][seq][hidden_size], pre-scattered (N = numDeepstack)
     * @param embedModel      for embedding subsequent tokens
     * @return generated token IDs for each batch element (excluding EOS)
     */
    public long[][] predict(float[][][] inputsEmbeds,
                            long[][] attentionMask,
                            long[][][] positionIds,
                            float[][][][] deepstackScattered,
                            Qwen3VlEmbedModel embedModel) throws OrtException {
        int batchSize = inputsEmbeds.length;

        // --- Prefill ---
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(inputsEmbeds, env));
        prefillInputs.put("attention_mask", createLongTensor2D(attentionMask));
        prefillInputs.put("position_ids", createLongTensor3D(positionIds));
        for (int d = 0; d < numDeepstack; d++) {
            prefillInputs.put(String.format(Locale.ROOT, "deepstack_scattered_%d", d),
                    ArrayUtil.createOnnxTensor(deepstackScattered[d], env));
        }

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

        OrtSession.Result prefillResult = session.run(prefillInputs, outputNames);

        float[][][] logits = (float[][][]) prefillResult.get("logits").get().getValue();
        float[][][][][] kvCache = new float[numLayers * 2][][][][];
        for (int i = 0; i < numLayers; i++) {
            kvCache[2 * i] = (float[][][][]) prefillResult
                    .get(String.format(Locale.ROOT, "present.%d.key", i)).get().getValue();
            kvCache[2 * i + 1] = (float[][][][]) prefillResult
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
        long maxPosition = max3D(positionIds);

        // --- Decode loop ---
        for (int step = 0; step < maxNewTokens - 1; step++) {
            if (ArrayUtil.allTrue(finished)) break;

            totalLen += 1;
            maxPosition += 1;

            float[][][] nextEmbeds = embedModel.predict(nextTokenIds);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(nextEmbeds, env));
            inputs.put("attention_mask", createLongTensor2D(ArrayUtil.ones(batchSize, totalLen)));

            // Position IDs for decode: all 3 dims share the same position (text token)
            long[][][] decodePosIds = new long[3][batchSize][1];
            for (int d = 0; d < 3; d++) {
                for (int b = 0; b < batchSize; b++) {
                    decodePosIds[d][b][0] = maxPosition;
                }
            }
            inputs.put("position_ids", createLongTensor3D(decodePosIds));

            // Deepstack: zeros for decode steps
            for (int d = 0; d < numDeepstack; d++) {
                float[][][] zeros = new float[batchSize][1][hiddenSize];
                inputs.put(String.format(Locale.ROOT, "deepstack_scattered_%d", d),
                        ArrayUtil.createOnnxTensor(zeros, env));
            }

            for (int i = 0; i < numLayers; i++) {
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.key", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i], env));
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.value", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i + 1], env));
            }

            OrtSession.Result stepResult = session.run(inputs, outputNames);

            logits = (float[][][]) stepResult.get("logits").get().getValue();
            for (int i = 0; i < numLayers; i++) {
                kvCache[2 * i] = (float[][][][]) stepResult
                        .get(String.format(Locale.ROOT, "present.%d.key", i)).get().getValue();
                kvCache[2 * i + 1] = (float[][][][]) stepResult
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

    private long max3D(long[][][] arr) {
        long max = Long.MIN_VALUE;
        for (long[][] d2 : arr) {
            for (long[] d1 : d2) {
                for (long v : d1) {
                    if (v > max) max = v;
                }
            }
        }
        return max;
    }

    /**
     * Create OnnxTensor from long[][] (2D).
     */
    private OnnxTensor createLongTensor2D(long[][] data) throws OrtException {
        return ArrayUtil.createOnnxTensor(data, env);
    }

    /**
     * Create OnnxTensor from long[][][] (3D) for MRoPE position_ids.
     */
    private OnnxTensor createLongTensor3D(long[][][] data) throws OrtException {
        int d0 = data.length;
        int d1 = data[0].length;
        int d2 = data[0][0].length;
        long[] flat = new long[d0 * d1 * d2];
        int idx = 0;
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                for (int k = 0; k < d2; k++) {
                    flat[idx++] = data[i][j][k];
                }
            }
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(flat), new long[]{d0, d1, d2});
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
