package io.github.flux.lightonocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Autoregressive decoder for LightOnOCR-2-1B with KV-cache.
 * Uses decoder_model_merged.onnx (Qwen3, 28 layers, 8 KV heads, head_dim 128).
 *
 * Inputs:  inputs_embeds [batch, seq_len, 1024], attention_mask [batch, total_len],
 *          past_key_values.{0-27}.{key,value} [batch, 8, past_len, 128]
 * Outputs: logits [batch, seq_len, 151936],
 *          present.{0-27}.{key,value} [batch, 8, total_len, 128]
 */
public class LightOnOcrDecoderModel implements AutoCloseable {

    public record PastKeyValueSpec(int numLayers, int numKvHeads, int headDim) {}

    private static final PastKeyValueSpec SPEC = new PastKeyValueSpec(28, 8, 128);
    private static final long STOP_TOKEN_ID = 151645L;   // <|im_end|>
    private static final long STOP_TOKEN_ID2 = 151643L;  // <|endoftext|>

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;
    private final int maxLength;

    public LightOnOcrDecoderModel(final String modelFile,
                                  final int gpuIndex,
                                  final OrtEnvironment env,
                                  final int maxLength) {
        this.maxLength = maxLength;
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
     * Autoregressive generation with KV-cache.
     *
     * @param imageFeatures [batch, num_patches, 1024] from vision encoder
     * @param inputIds      [batch, seq_len] token IDs (with image_pad placeholders)
     * @param inputsEmbeds  [batch, seq_len, 1024] from embed model
     * @param attentionMask [batch, seq_len] initial attention mask (all 1s)
     * @param embedModel    for embedding generated tokens
     * @return generated token IDs for each batch item
     */
    public long[][] predict(float[][][] imageFeatures,
                            long[][] inputIds,
                            float[][][] inputsEmbeds,
                            long[][] attentionMask,
                            LightOnOcrEmbedModel embedModel) throws OrtException {
        int batchSize = inputsEmbeds.length;
        int numLayers = SPEC.numLayers();
        int numKvHeads = SPEC.numKvHeads();
        int headDim = SPEC.headDim();

        // --- Prefill ---
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(inputsEmbeds, env));
        prefillInputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));

        // Empty KV cache [batch, num_kv_heads, 0, head_dim]
        for (int i = 0; i < numLayers; i++) {
            float[][][][] emptyKv = new float[batchSize][numKvHeads][0][headDim];
            OnnxTensor emptyTensor = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batchSize, numKvHeads, 0, headDim});
            prefillInputs.put(String.format(Locale.ROOT, "past_key_values.%d.key", i), emptyTensor);
            // reuse same empty buffer for value (separate tensor)
            OnnxTensor emptyTensorV = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batchSize, numKvHeads, 0, headDim});
            prefillInputs.put(String.format(Locale.ROOT, "past_key_values.%d.value", i), emptyTensorV);
        }

        OrtSession.Result prefillResult = session.run(prefillInputs, outputNames);

        // Extract logits and KV cache
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

        // First generated token
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

        int currLen = inputIds[0].length;
        boolean[] finished = new boolean[batchSize];

        // Check if first token is already EOS
        for (int b = 0; b < batchSize; b++) {
            if (firstTokens[b] == STOP_TOKEN_ID || firstTokens[b] == STOP_TOKEN_ID2) {
                finished[b] = true;
            }
        }

        // --- Decode loop ---
        for (int step = 0; step < maxLength - 1; step++) {
            if (ArrayUtil.allTrue(finished)) {
                break;
            }

            currLen += 1;

            // Embed next token
            float[][][] nextEmbeds = embedModel.predict(nextTokenIds);

            // Build decoder inputs
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(nextEmbeds, env));
            inputs.put("attention_mask", ArrayUtil.createOnnxTensor(
                    ArrayUtil.ones(batchSize, currLen), env));

            for (int i = 0; i < numLayers; i++) {
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.key", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i], env));
                inputs.put(String.format(Locale.ROOT, "past_key_values.%d.value", i),
                        ArrayUtil.createOnnxTensor(kvCache[2 * i + 1], env));
            }

            // Run decoder
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

            // Pick next tokens
            long[][] nextIds = new long[batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                if (finished[b]) {
                    continue;
                }
                float[] lastLogit = logits[b][0];
                long nextToken = ArrayUtil.argmax(lastLogit);
                nextIds[b][0] = nextToken;
                generatedTokens[b] = ArrayUtil.concat(generatedTokens[b], new long[]{nextToken});
                if (nextToken == STOP_TOKEN_ID || nextToken == STOP_TOKEN_ID2) {
                    finished[b] = true;
                }
            }
            nextTokenIds = nextIds;
        }

        return generatedTokens;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
