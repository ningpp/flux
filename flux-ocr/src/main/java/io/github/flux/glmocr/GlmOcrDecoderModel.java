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
 * LLM decoder for GLM-OCR using separate prefill and decode models.
 * 
 * The prefill model has dynamic sequence length and is used for initial prompt processing.
 * The decode model has fixed seq_len=1 and is used for autoregressive token generation.
 * 
 * GLM-OCR specific: Uses 3D position_ids [3, batch, seq_len] for rotary embeddings.
 */
public class GlmOcrDecoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession prefillSession;
    private final OrtSession decodeSession;
    private final int maxLength;

    // Model architecture constants
    private static final int NUM_LAYERS = 16;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = 128;

    /**
     * Create decoder model with separate prefill and decode sessions.
     *
     * @param prefillModelFile path to llm_prefill.onnx
     * @param decodeModelFile path to llm_decode.onnx
     * @param gpuIndex GPU index (-1 for CPU)
     * @param env ONNX Runtime environment
     * @param maxLength maximum generation length
     */
    public GlmOcrDecoderModel(final String prefillModelFile,
                              final String decodeModelFile,
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
            this.prefillSession = env.createSession(prefillModelFile, options);
            this.decodeSession = env.createSession(decodeModelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Generate tokens autoregressively.
     *
     * @param imageFeatures vision encoder output [batch, num_patches, hidden] (unused, for API compat)
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

        // Create attention mask [batch, seq_len]
        long[][] attentionMask = ArrayUtil.ones(batchSize, seqLen);

        // Create position_ids [3, batch, seq_len] for GLM-OCR's rotary embeddings
        long[][][] positionIds = new long[3][batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                positionIds[0][b][i] = i;  // position
                positionIds[1][b][i] = 0;  // block_position
                positionIds[2][b][i] = 0;  // is_image (0 for text)
            }
        }

        // Prefill inputs (no KV cache needed for prefill model)
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", createFloatTensor3D(inputsEmbeds));
        prefillInputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));
        prefillInputs.put("position_ids", createPositionIdsTensor(positionIds));

        // Prefill phase - get initial logits and KV cache
        Result prefillResult = prefillSession.run(prefillInputs);
        float[][][] logits = (float[][][]) prefillResult.get(0).getValue();

        // Extract KV cache from prefill output: present_key_0, present_value_0, ...
        float[][][][][] pkvs = new float[NUM_LAYERS * 2][][][][];
        for (int i = 0; i < NUM_LAYERS; i++) {
            pkvs[2 * i] = (float[][][][]) prefillResult.get(2 * i + 1).getValue();      // present_key_i
            pkvs[2 * i + 1] = (float[][][][]) prefillResult.get(2 * i + 2).getValue();  // present_value_i
        }

        OnnxUtil.closeTensors(prefillInputs);

        // Get first predicted token from prefill
        long eosTokenId = 151643L;  // GLM-OCR EOS token
        long start = ArrayUtil.argmax(logits[0][logits[0].length - 1]);

        long[][] generatedTokens = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            generatedTokens[i] = new long[]{start};
        }

        long[][] nextTokenIds = new long[batchSize][1];
        for (int i = 0; i < batchSize; i++) {
            nextTokenIds[i][0] = start;
        }

        int currLen = seqLen;
        boolean[] finished = new boolean[batchSize];

        // Check if already finished
        for (int i = 0; i < batchSize; i++) {
            if (start == eosTokenId) {
                finished[i] = true;
            }
        }

        IOUtil.close(prefillResult);

        // Autoregressive decode loop
        for (int step = 0; step < maxLength; step++) {
            if (ArrayUtil.allTrue(finished)) {
                break;
            }
            System.out.println(String.format("%6d", step) + " ".repeat(11) + java.time.LocalDateTime.now());

            currLen += 1;

            // Get embeddings for next tokens
            float[][][] nextEmbed = embedModel.predict(nextTokenIds);

            Map<String, OnnxTensor> decodeInputs = new HashMap<>();
            decodeInputs.put("inputs_embeds", createFloatTensor3D(nextEmbed));

            // Update attention mask for current length
            decodeInputs.put("attention_mask", ArrayUtil.createOnnxTensor(ArrayUtil.ones(batchSize, currLen), env));

            // Position IDs for single token decode: [3, batch, 1]
            long[][][] nextPosIds = new long[3][batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                nextPosIds[0][b][0] = currLen - 1;  // position
                nextPosIds[1][b][0] = 0;            // block_position
                nextPosIds[2][b][0] = 0;            // is_image
            }
            decodeInputs.put("position_ids", createPositionIdsTensor(nextPosIds));

            // Add KV cache from previous step
            for (int j = 0; j < NUM_LAYERS; j++) {
                decodeInputs.put("past_key_" + j, createFloatTensor4D(pkvs[2 * j]));
                decodeInputs.put("past_value_" + j, createFloatTensor4D(pkvs[2 * j + 1]));
            }

            Result stepOut = decodeSession.run(decodeInputs);
            logits = (float[][][]) stepOut.get(0).getValue();

            // Update KV cache: decode model outputs present_key_i, present_value_i
            for (int i = 0; i < NUM_LAYERS; i++) {
                pkvs[2 * i] = (float[][][][]) stepOut.get(2 * i + 1).getValue();
                pkvs[2 * i + 1] = (float[][][][]) stepOut.get(2 * i + 2).getValue();
            }

            IOUtil.close(stepOut);
            OnnxUtil.closeTensors(decodeInputs);

            // Get next token for each batch
            long[][] nextIds = new long[batchSize][1];
            for (int j = 0; j < batchSize; j++) {
                if (finished[j]) {
                    nextIds[j][0] = eosTokenId;  // Pad with EOS
                    continue;
                }
                float[] lastLogit = logits[j][0];
                long nextToken = ArrayUtil.argmax(lastLogit);
                nextIds[j][0] = nextToken;

                generatedTokens[j] = ArrayUtil.concat(generatedTokens[j], new long[]{nextToken});
                if (nextToken == eosTokenId) {
                    finished[j] = true;
                }
            }
            nextTokenIds = nextIds;
        }

        return generatedTokens;
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
        int d4 = data[0][0][0].length;
        float[] flat = ArrayUtil.flat(data);
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{d1, d2, d3, d4});
    }

    /**
     * Create position_ids tensor with shape [3, batch, seq_len]
     */
    private OnnxTensor createPositionIdsTensor(long[][][] positionIds) throws OrtException {
        int d1 = positionIds.length;        // 3
        int d2 = positionIds[0].length;     // batch
        int d3 = positionIds[0][0].length;  // seq_len

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
        prefillSession.close();
        decodeSession.close();
    }
}
