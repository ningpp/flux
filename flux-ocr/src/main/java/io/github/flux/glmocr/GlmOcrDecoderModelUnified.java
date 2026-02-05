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

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * LLM decoder for GLM-OCR using unified model (llm_unified.onnx).
 * 
 * The unified model has fixed seq_len=1 and processes one token at a time.
 * This uses less memory than separate prefill/decode models since only one
 * model is loaded, but prefill is slower due to token-by-token processing.
 *
 * GLM-OCR specific: Uses 3D position_ids [3, batch, seq_len] for rotary embeddings.
 */
public class GlmOcrDecoderModelUnified implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int maxLength;

    // Model architecture constants
    private static final int NUM_LAYERS = 16;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = 128;
    private static final int HIDDEN_SIZE = 1536;

    // Stop tokens
    private static final long EOS_TOKEN_ID = 59246L;   // <|endoftext|>
    private static final long USER_TOKEN_ID = 59253L;  // <|user|>

    /**
     * Create decoder model with unified session.
     *
     * @param modelFile path to llm_unified.onnx
     * @param gpuIndex GPU index (-1 for CPU)
     * @param env ONNX Runtime environment
     * @param maxLength maximum generation length
     */
    public GlmOcrDecoderModelUnified(final String modelFile,
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
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Generate tokens autoregressively.
     *
     * @param imageFeatures vision encoder output (unused, for API compat)
     * @param inputIds initial input token IDs [batch][seq_len]
     * @param inputsEmbeds initial embeddings with image features merged [batch][seq_len][hidden]
     * @param embedModel embedding model for decoding new tokens
     * @param ndManager NDArray manager
     * @return generated token IDs for each batch item
     */
    public long[][] predict(float[][][] imageFeatures,
                            long[][] inputIds,
                            float[][][] inputsEmbeds,
                            GlmOcrEmbedModel embedModel,
                            NDManager ndManager) throws OrtException {
        int batchSize = inputsEmbeds.length;
        int seqLen = inputsEmbeds[0].length;

        // Initialize empty KV cache - will grow during prefill
        float[][][][][] pkvs = null;

        // Process prefill token by token
        for (int pos = 0; pos < seqLen; pos++) {
            // Extract single token embedding [batch, 1, hidden]
            float[][][] singleEmbed = new float[batchSize][1][HIDDEN_SIZE];
            for (int b = 0; b < batchSize; b++) {
                singleEmbed[b][0] = inputsEmbeds[b][pos];
            }

            // Run single token through model
            pkvs = runSingleToken(singleEmbed, pos, pkvs);
        }

        // Get first generated token from final prefill step
        float[][][] lastEmbed = new float[batchSize][1][HIDDEN_SIZE];
        for (int b = 0; b < batchSize; b++) {
            lastEmbed[b][0] = inputsEmbeds[b][seqLen - 1];
        }
        
        // Run final prefill step to get logits
        Map<String, OnnxTensor> inputs = buildInputs(lastEmbed, seqLen - 1, pkvs);
        Result result = session.run(inputs);
        float[][][] logits = (float[][][]) result.get(0).getValue();
        
        // Update KV cache
        pkvs = extractKVCache(result);
        IOUtil.close(result);
        OnnxUtil.closeTensors(inputs);

        long start = ArrayUtil.argmax(logits[0][0]);

        long[][] generatedTokens = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            generatedTokens[i] = new long[]{start};
        }

        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < batchSize; i++) {
            if (start == EOS_TOKEN_ID || start == USER_TOKEN_ID) {
                finished[i] = true;
            }
        }

        int currPos = seqLen;

        // Autoregressive decode loop
        for (int step = 0; step < maxLength; step++) {
            if (ArrayUtil.allTrue(finished)) {
                break;
            }

            // Get embeddings for next tokens
            long[][] nextTokenIds = new long[batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                nextTokenIds[b][0] = generatedTokens[b][generatedTokens[b].length - 1];
            }
            float[][][] nextEmbed = embedModel.predict(nextTokenIds);

            // Run decode step
            inputs = buildInputs(nextEmbed, currPos, pkvs);
            result = session.run(inputs);
            logits = (float[][][]) result.get(0).getValue();
            pkvs = extractKVCache(result);
            IOUtil.close(result);
            OnnxUtil.closeTensors(inputs);

            currPos++;

            // Get next token for each batch
            for (int j = 0; j < batchSize; j++) {
                if (finished[j]) {
                    continue;
                }
                long nextToken = ArrayUtil.argmax(logits[j][0]);
                generatedTokens[j] = ArrayUtil.concat(generatedTokens[j], new long[]{nextToken});
                if (nextToken == EOS_TOKEN_ID || nextToken == USER_TOKEN_ID) {
                    finished[j] = true;
                }
            }
        }

        return generatedTokens;
    }

    /**
     * Run single token through unified model, updating KV cache.
     */
    private float[][][][][] runSingleToken(float[][][] embed, int pos, float[][][][][] prevPkvs) 
            throws OrtException {
        Map<String, OnnxTensor> inputs = buildInputs(embed, pos, prevPkvs);
        Result result = session.run(inputs);
        float[][][][][] newPkvs = extractKVCache(result);
        IOUtil.close(result);
        OnnxUtil.closeTensors(inputs);
        return newPkvs;
    }

    /**
     * Build inputs for unified model.
     */
    private Map<String, OnnxTensor> buildInputs(float[][][] embed, int pos, float[][][][][] pkvs) 
            throws OrtException {
        int batchSize = embed.length;
        int cacheLen = (pkvs == null) ? 0 : pkvs[0][0][0].length;
        int totalLen = cacheLen + 1;

        Map<String, OnnxTensor> inputs = new HashMap<>();

        // inputs_embeds: [batch, 1, hidden]
        inputs.put("inputs_embeds", createFloatTensor3D(embed));

        // attention_mask: [batch, total_len]
        long[][] attentionMask = ArrayUtil.ones(batchSize, totalLen);
        inputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));

        // position_ids: [3, batch, 1]
        long[][][] positionIds = new long[3][batchSize][1];
        for (int b = 0; b < batchSize; b++) {
            positionIds[0][b][0] = pos;  // position
            positionIds[1][b][0] = pos;  // same for GLM-OCR
            positionIds[2][b][0] = pos;  // same for GLM-OCR
        }
        inputs.put("position_ids", createPositionIdsTensor(positionIds));

        // KV cache
        if (pkvs == null) {
            // Empty cache for first token
            for (int i = 0; i < NUM_LAYERS; i++) {
                float[][][][] emptyKey = new float[batchSize][NUM_KV_HEADS][0][HEAD_DIM];
                float[][][][] emptyValue = new float[batchSize][NUM_KV_HEADS][0][HEAD_DIM];
                inputs.put("past_key_" + i, createFloatTensor4D(emptyKey));
                inputs.put("past_value_" + i, createFloatTensor4D(emptyValue));
            }
        } else {
            for (int i = 0; i < NUM_LAYERS; i++) {
                inputs.put("past_key_" + i, createFloatTensor4D(pkvs[2 * i]));
                inputs.put("past_value_" + i, createFloatTensor4D(pkvs[2 * i + 1]));
            }
        }

        return inputs;
    }

    /**
     * Extract KV cache from model output.
     */
    private float[][][][][] extractKVCache(Result result) throws OrtException {
        float[][][][][] pkvs = new float[NUM_LAYERS * 2][][][][];
        for (int i = 0; i < NUM_LAYERS; i++) {
            // Output order: logits, then key/value pairs
            pkvs[2 * i] = (float[][][][]) result.get(2 * i + 1).getValue();
            pkvs[2 * i + 1] = (float[][][][]) result.get(2 * i + 2).getValue();
        }
        return pkvs;
    }

    /**
     * Create a 3D float tensor.
     */
    private OnnxTensor createFloatTensor3D(float[][][] data) throws OrtException {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        float[] flat = ArrayUtil.flat(data);
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{d1, d2, d3});
    }

    /**
     * Create a 4D float tensor.
     */
    private OnnxTensor createFloatTensor4D(float[][][][] data) throws OrtException {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = (d3 > 0) ? data[0][0][0].length : HEAD_DIM;
        float[] flat = ArrayUtil.flat(data);
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{d1, d2, d3, d4});
    }

    /**
     * Create position_ids tensor with shape [3, batch, seq_len]
     */
    private OnnxTensor createPositionIdsTensor(long[][][] positionIds) throws OrtException {
        int d1 = positionIds.length;
        int d2 = positionIds[0].length;
        int d3 = positionIds[0][0].length;

        long[] flat = new long[d1 * d2 * d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    flat[idx++] = positionIds[i][j][k];
                }
            }
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(flat), new long[]{d1, d2, d3});
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
