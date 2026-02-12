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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
public class GlmOcrDecoderModelUnified implements GlmOcrDecoder {

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

    // MRoPE / spatial merge constants
    private static final int MERGE_SIZE = 2;
    private static final long IMAGE_TOKEN_ID = 59280L;

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
     * @param imageGridThw image grid dimensions [t, h, w] in patch units
     * @param embedModel embedding model for decoding new tokens
     * @param ndManager NDArray manager
     * @return generated token IDs for each batch item
     */
    @Override
    public long[][] predict(float[][][] imageFeatures,
                            long[][] inputIds,
                            float[][][] inputsEmbeds,
                            int[] imageGridThw,
                            GlmOcrEmbedModel embedModel,
                            NDManager ndManager) throws OrtException {
        int batchSize = inputsEmbeds.length;
        int seqLen = inputsEmbeds[0].length;

        // Pre-compute 3D MRoPE position_ids [3][1][seqLen]
        long[][][] prefillPositionIds = computePositionIds(inputIds[0], imageGridThw);

        // Track the last position for decode continuation
        long lastPos = 0;
        for (int d = 0; d < 3; d++) {
            lastPos = Math.max(lastPos, prefillPositionIds[d][0][seqLen - 1]);
        }

        // Initialize empty KV cache - will grow during prefill
        OnnxTensor[] pkvTensors = null;

        // Process prefill token by token (tokens 0..seqLen-2 for KV cache building)
        for (int pos = 0; pos < seqLen - 1; pos++) {
            float[][][] singleEmbed = new float[batchSize][1][HIDDEN_SIZE];
            for (int b = 0; b < batchSize; b++) {
                singleEmbed[b][0] = inputsEmbeds[b][pos];
            }

            long[] dimPos = new long[]{prefillPositionIds[0][0][pos],
                                       prefillPositionIds[1][0][pos],
                                       prefillPositionIds[2][0][pos]};
            pkvTensors = runSingleTokenTensor(ArrayUtil.createOnnxTensor(singleEmbed, env), dimPos, pkvTensors);
        }

        // Run final prefill token to get logits
        float[][][] lastEmbed = new float[batchSize][1][HIDDEN_SIZE];
        for (int b = 0; b < batchSize; b++) {
            lastEmbed[b][0] = inputsEmbeds[b][seqLen - 1];
        }

        long[] lastDimPos = new long[]{prefillPositionIds[0][0][seqLen - 1],
                                       prefillPositionIds[1][0][seqLen - 1],
                                       prefillPositionIds[2][0][seqLen - 1]};
        Map<String, OnnxTensor> inputs = buildInputs(ArrayUtil.createOnnxTensor(lastEmbed, env), lastDimPos, pkvTensors);
        Result result = session.run(inputs);
        float[][][] logits = (float[][][]) result.get(0).getValue();

        // Update KV cache
        pkvTensors = extractKVCacheTensor(result);
        // IOUtil.close(result);
        // OnnxUtil.closeTensors(inputs);

        long start = ArrayUtil.argmax(logits[0][0]);

        // Initialize generatedTokens with input tokens + first generated token
        long[][] generatedTokens = new long[inputIds.length][];
        for (int i = 0; i < inputIds.length; i++) {
            generatedTokens[i] = ArrayUtil.concat(ArrayUtil.clone(inputIds[i]), new long[]{start});
        }

        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < batchSize; i++) {
            if (start == EOS_TOKEN_ID || start == USER_TOKEN_ID) {
                finished[i] = true;
            }
        }

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
            OnnxTensor nextEmbed = embedModel.predictTensor(nextTokenIds);

            // For decode, all 3 dims get the same sequential position
            long decodePos = lastPos + 1 + step;
            long[] decodeDimPos = new long[]{decodePos, decodePos, decodePos};
            inputs = buildInputs(nextEmbed, decodeDimPos, pkvTensors);
            result = session.run(inputs);
            // OnnxUtil.closeTensors(inputs);
            logits = (float[][][]) result.get(0).getValue();
            for (OnnxTensor pkvTensor : pkvTensors) {
                IOUtil.close(pkvTensor);
            }
            pkvTensors = extractKVCacheTensor(result);
            // IOUtil.close(result);
            OnnxUtil.closeTensors(inputs);

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

        long[][] resultTokens = new long[inputIds.length][];
        int index = 0;
        for (long[] ids : inputIds) {
            long[] tokens = new long[generatedTokens[index].length-ids.length];
            System.arraycopy(generatedTokens[index], ids.length, tokens, 0, generatedTokens[index].length-ids.length);
            resultTokens[index] = tokens;
            index++;
        }
        for (OnnxTensor pkvTensor : pkvTensors) {
            IOUtil.close(pkvTensor);
        }
        return resultTokens;
    }

    /**
     * Run single token through unified model, updating KV cache.
     */
    private OnnxTensor[] runSingleTokenTensor(OnnxTensor embed, long[] dimPos, OnnxTensor[] prevPkvs)
            throws OrtException {
        Map<String, OnnxTensor> inputs = buildInputs(embed, dimPos, prevPkvs);
        Result result = session.run(inputs);
        OnnxTensor[] newPkvs = extractKVCacheTensor(result);
        OnnxUtil.closeTensors(inputs);
        return newPkvs;
    }

    /**
     * Build inputs for unified model.
     *
     * @param embed    single token embedding [batch, 1, hidden]
     * @param dimPos   position for each of the 3 MRoPE dims [temporal, height, width]
     * @param pkvs     previous KV cache (null for first token)
     */
    private Map<String, OnnxTensor> buildInputs(OnnxTensor embed, long[] dimPos, OnnxTensor[] pkvs)
            throws OrtException {
        long[] embedShape = embed.getInfo().getShape();
        int batchSize = (int) embedShape[0];
        int cacheLen = (pkvs == null) ? 0 : (int) pkvs[0].getInfo().getShape()[2];
        int totalLen = cacheLen + 1;

        Map<String, OnnxTensor> inputs = new HashMap<>();

        // inputs_embeds: [batch, 1, hidden]
        inputs.put("inputs_embeds", embed);

        // attention_mask: [batch, total_len]
        long[][] attentionMask = ArrayUtil.ones(batchSize, totalLen);
        inputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));

        // position_ids: [3, batch, 1] with proper per-dim MRoPE positions
        long[][][] positionIds = new long[3][batchSize][1];
        for (int b = 0; b < batchSize; b++) {
            positionIds[0][b][0] = dimPos[0];
            positionIds[1][b][0] = dimPos[1];
            positionIds[2][b][0] = dimPos[2];
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
                inputs.put("past_key_" + i, pkvs[2 * i]);
                inputs.put("past_value_" + i, pkvs[2 * i + 1]);
            }
        }

        return inputs;
    }

    /**
     * Extract KV cache from model output.
     */
    private OnnxTensor[] extractKVCacheTensor(Result result) throws OrtException {
        OnnxTensor[] pkvs = new OnnxTensor[NUM_LAYERS * 2];
        for (int i = 0; i < NUM_LAYERS; i++) {
            // Output order: logits, then key/value pairs
            pkvs[2 * i] = (OnnxTensor) result.get(2 * i + 1);
            pkvs[2 * i + 1] = (OnnxTensor) result.get(2 * i + 2);
        }
        return pkvs;
    }

    /**
     * Create a 4D float tensor.
     */
    private OnnxTensor createFloatTensor4D(float[][][][] data) throws OrtException {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = HEAD_DIM;
        // Handle empty cache (d3 == 0)
        if (d3 == 0) {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[0]), new long[]{d1, d2, 0, d4});
        }
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

    /**
     * Compute 3D MRoPE position_ids for GLM-OCR.
     * Text tokens: all 3 dims = sequential.
     * Vision tokens: dim0=temporal, dim1=height, dim2=width with spatial layout.
     *
     * @param inputIds     flat input token IDs for single batch
     * @param imageGridThw [t, h, w] in patch units for the image
     * @return position_ids [3][1][seqLen]
     */
    private long[][][] computePositionIds(long[] inputIds, int[] imageGridThw) {
        int seqLen = inputIds.length;
        List<long[][]> segments = new ArrayList<>();
        int st = 0;
        boolean imageProcessed = false;

        for (int i = 0; i < seqLen; i++) {
            if (inputIds[i] == IMAGE_TOKEN_ID && !imageProcessed) {
                int imgStart = i;
                int textLen = imgStart - st;

                // Find end of image token region first
                int imgEnd = imgStart;
                while (imgEnd < seqLen && inputIds[imgEnd] == IMAGE_TOKEN_ID) imgEnd++;
                int actualVisTokens = imgEnd - imgStart;

                int t = imageGridThw[0];
                int h = imageGridThw[1];
                int w = imageGridThw[2];

                int llmGridT = t;
                int llmGridH = h / MERGE_SIZE;
                int llmGridW = w / MERGE_SIZE;

                long stIdx = segments.isEmpty() ? 0 : maxOfLastSegment(segments) + 1;

                // Text positions before image
                if (textLen > 0) {
                    long[][] textPos = new long[3][textLen];
                    for (int d = 0; d < 3; d++) {
                        for (int p = 0; p < textLen; p++) {
                            textPos[d][p] = stIdx + p;
                        }
                    }
                    segments.add(textPos);
                }

                // Vision positions: T/H/W spatial grid, capped at actual token count
                long[][] visPos = new long[3][actualVisTokens];
                int vIdx = 0;
                for (int gt = 0; gt < llmGridT && vIdx < actualVisTokens; gt++) {
                    for (int gh = 0; gh < llmGridH && vIdx < actualVisTokens; gh++) {
                        for (int gw = 0; gw < llmGridW && vIdx < actualVisTokens; gw++) {
                            visPos[0][vIdx] = gt + textLen + stIdx;
                            visPos[1][vIdx] = gh + textLen + stIdx;
                            visPos[2][vIdx] = gw + textLen + stIdx;
                            vIdx++;
                        }
                    }
                }
                segments.add(visPos);

                st = imgEnd;
                i = imgEnd - 1;
                imageProcessed = true;
            }
        }

        // Remaining text tokens after image
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

        // Concatenate segments into [3, 1, seqLen]
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

    @Override
    public void close() throws Exception {
        session.close();
    }
}
