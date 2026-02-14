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
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
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
 * LLM decoder for GLM-OCR using separate prefill and decode models.
 * 
 * The prefill model has dynamic sequence length and is used for initial prompt processing.
 * The decode model has fixed seq_len=1 and is used for autoregressive token generation.
 * 
 * GLM-OCR specific: Uses 3D position_ids [3, batch, seq_len] for rotary embeddings.
 */
public class GlmOcrDecoderModel implements GlmOcrDecoder {

    private final OrtEnvironment env;
    private final OrtSession prefillSession;
    private final OrtSession decodeSession;
    private final int maxLength;

    // Model architecture constants
    private static final int NUM_LAYERS = 16;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = 128;

    // Spatial merge size for position_ids computation
    private static final int MERGE_SIZE = 2;

    // Image token ID for GLM-OCR
    private static final long IMAGE_TOKEN_ID = 59280L;

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
                // Single thread for GPU - let runtime handle parallelism
                options.setIntraOpNumThreads(1);
                options.setInterOpNumThreads(1);
            }
            options.setOptimizationLevel(OptLevel.ALL_OPT);
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

        // Create attention mask [batch, seq_len]
        long[][] attentionMask = ArrayUtil.ones(batchSize, seqLen);

        // Compute proper 3D position_ids [3, batch, seq_len] for GLM-OCR's MRoPE
        long[][][] positionIds = computePositionIds(inputIds[0], imageGridThw);
        // The last position in prefill determines where decode positions start
        long lastPos = 0;
        for (int d = 0; d < 3; d++) {
            lastPos = Math.max(lastPos, positionIds[d][0][seqLen - 1]);
        }

        // Prefill inputs (no KV cache needed for prefill model)
        Map<String, OnnxTensor> prefillInputs = new HashMap<>();
        prefillInputs.put("inputs_embeds", createFloatTensor3D(inputsEmbeds));
        prefillInputs.put("attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env));
        prefillInputs.put("position_ids", createPositionIdsTensor(positionIds));

        // Prefill phase - get initial logits and KV cache
        Result prefillResult = prefillSession.run(prefillInputs);
        float[][][] logits = (float[][][]) prefillResult.get(0).getValue();

        // Extract KV cache from prefill output as OnnxTensor[] (zero-copy)
        OnnxTensor[] pkvTensors = new OnnxTensor[NUM_LAYERS * 2];
        for (int i = 0; i < NUM_LAYERS; i++) {
            pkvTensors[2 * i] = (OnnxTensor) prefillResult.get(2 * i + 1);      // present_key_i
            pkvTensors[2 * i + 1] = (OnnxTensor) prefillResult.get(2 * i + 2);  // present_value_i
        }

        OnnxUtil.closeTensors(prefillInputs);

        // Get first predicted token from prefill
        long eosTokenId = 59246L;  // GLM-OCR EOS token <|endoftext|>
        long userTokenId = 59253L;  // <|user|> - also signals end of response
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
            if (start == eosTokenId || start == userTokenId) {
                finished[i] = true;
            }
        }

        // IOUtil.close(prefillResult);

        // Track previous result to close it properly
        Result prevResult = null;

        // Autoregressive decode loop
        for (int step = 0; step < maxLength; step++) {
            if (ArrayUtil.allTrue(finished)) {
                break;
            }
            currLen += 1;

            // Get embeddings for next tokens
            OnnxTensor nextEmbed = embedModel.predictTensor(nextTokenIds);

            Map<String, OnnxTensor> decodeInputs = new HashMap<>();
            decodeInputs.put("inputs_embeds", nextEmbed);

            // Update attention mask for current length
            decodeInputs.put("attention_mask", ArrayUtil.createOnnxTensor(ArrayUtil.ones(batchSize, currLen), env));

            // Position IDs for single token decode: [3, batch, 1]
            // For MRoPE, all 3 dims get the same sequential position after prefill
            long[][][] nextPosIds = new long[3][batchSize][1];
            long decodePos = lastPos + (currLen - seqLen);
            for (int b = 0; b < batchSize; b++) {
                nextPosIds[0][b][0] = decodePos;
                nextPosIds[1][b][0] = decodePos;
                nextPosIds[2][b][0] = decodePos;
            }
            decodeInputs.put("position_ids", createPositionIdsTensor(nextPosIds));

            // Add KV cache from previous step (pass tensors directly, no copy)
            for (int j = 0; j < NUM_LAYERS; j++) {
                decodeInputs.put("past_key_" + j, pkvTensors[2 * j]);
                decodeInputs.put("past_value_" + j, pkvTensors[2 * j + 1]);
            }

            Result stepOut = decodeSession.run(decodeInputs);
            logits = (float[][][]) stepOut.get(0).getValue();

            // Update KV cache: decode model outputs present_key_i, present_value_i
            // Close previous result (which owns old KV cache tensors) first
            if (prevResult != null) {
                IOUtil.close(prevResult);
            }
            // Extract new KV cache tensors from current result
            pkvTensors = new OnnxTensor[NUM_LAYERS * 2];
            for (int i = 0; i < NUM_LAYERS; i++) {
                pkvTensors[2 * i] = (OnnxTensor) stepOut.get(2 * i + 1);
                pkvTensors[2 * i + 1] = (OnnxTensor) stepOut.get(2 * i + 2);
            }
            prevResult = stepOut;  // Keep reference to close later
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
                if (nextToken == eosTokenId || nextToken == userTokenId) {
                    finished[j] = true;
                }
            }
            nextTokenIds = nextIds;
        }

        // Clean up: close final KV cache result
        if (prevResult != null) {
            IOUtil.close(prevResult);
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

    /**
     * Compute 3D MRoPE position_ids for GLM-OCR (same as Qwen3-VL).
     * Text tokens: all 3 dims = sequential.
     * Vision tokens: dim0=temporal, dim1=height, dim2=width with spatial layout.
     *
     * @param inputIds     flat input token IDs for a single batch item
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
        prefillSession.close();
        decodeSession.close();
    }
}
