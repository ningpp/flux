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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
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
 * Unified LLM decoder model for GLM-OCR.
 * Handles both prefill and decode phases with KV-cache.
 * 
 * GLM-OCR specific: Uses 3D position_ids [3, batch, seq_len] for rotary embeddings.
 */
public class GlmOcrDecoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int maxLength;
    
    // Model architecture constants
    private static final int NUM_LAYERS = 16;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = 128;

    public GlmOcrDecoderModel(final String modelFile,
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
     * @param imageFeatures vision encoder output [batch, num_patches, hidden]
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
        int batchSize = imageFeatures.length;
        int seqLen = inputIds[0].length;

        // Create attention mask [batch, seq_len]
        long[][] attentionMask = ArrayUtil.ones(batchSize, seqLen);

        // Create position_ids [3, batch, seq_len] for GLM-OCR's rotary embeddings
        // [position, block_position, is_image] format
        long[][][] positionIds = new long[3][batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                positionIds[0][b][i] = i;  // position
                positionIds[1][b][i] = 0;  // block_position
                positionIds[2][b][i] = 0;  // is_image (0 for text)
            }
        }

        // Initialize empty KV cache
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(inputsEmbeds, env));
        prefillInputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));
        prefillInputs.put("position_ids", createPositionIdsTensor(positionIds));
        
        // Add empty KV cache for prefill
        for (int i = 0; i < NUM_LAYERS; i++) {
            NDArray emptyKV = ndManager.zeros(new Shape(batchSize, NUM_KV_HEADS, 0, HEAD_DIM), DataType.FLOAT32);
            FloatBuffer buffer = emptyKV.toByteBuffer().asFloatBuffer();
            long[] shape = emptyKV.getShape().getShape();
            prefillInputs.put("past_key_" + i, OnnxTensor.createTensor(env, buffer, shape));
            prefillInputs.put("past_value_" + i, OnnxTensor.createTensor(env, buffer, shape));
        }

        // Prefill phase
        Result prefillResult = session.run(prefillInputs);
        float[][][] logits = (float[][][]) prefillResult.get(0).getValue();
        
        // Extract KV cache from prefill output
        float[][][][][] pkvs = new float[NUM_LAYERS * 2][][][][];
        for (int i = 0; i < NUM_LAYERS * 2; i++) {
            pkvs[i] = (float[][][][]) prefillResult.get(i + 1).getValue();
        }

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

        // Autoregressive decode loop
        for (int step = 0; step < maxLength; step++) {
            if (ArrayUtil.allTrue(finished)) {
                break;
            }

            currLen += 1;
            
            // Get embeddings for next tokens
            float[][][] nextEmbed = embedModel.predict(nextTokenIds);

            Map<String, OnnxTensor> decodeInputs = new HashMap<>();
            decodeInputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(nextEmbed, env));
            
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
                decodeInputs.put("past_key_" + j, ArrayUtil.createOnnxTensor(pkvs[2 * j], env));
                decodeInputs.put("past_value_" + j, ArrayUtil.createOnnxTensor(pkvs[2 * j + 1], env));
            }

            Result stepOut = session.run(decodeInputs);
            logits = (float[][][]) stepOut.get(0).getValue();
            
            // Update KV cache
            for (int o = 0; o < NUM_LAYERS * 2; o++) {
                pkvs[o] = (float[][][][]) stepOut.get(o + 1).getValue();
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
        session.close();
    }
}
