// this code is convert from  https://github.com/breezedeus/Pix2Text
// Pix2Text IS Licensed under the MIT License
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
package io.github.flux.formula.pix2text;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Pix2TextDecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int maxLength;
    private final long padTokenId;
    private final long eosTokenId;
    private final long decoderStartTokenId;
    private final HuggingFaceTokenizer tokenizer;

    public Pix2TextDecoderModel(final String modelFile,
                                final int gpuIndex,
                                final OrtEnvironment env,
                                int maxLength,
                                long padTokenId,
                                long eosTokenId,
                                long decoderStartTokenId,
                                HuggingFaceTokenizer tokenizer) {
        this.maxLength = maxLength;
        this.padTokenId = padTokenId;
        this.eosTokenId = eosTokenId;
        this.tokenizer = tokenizer;
        this.decoderStartTokenId = decoderStartTokenId;
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public List<TextResult> batchPredict(float[][][] encodeResultFloats, MatManager matManager) throws OrtException {
        int encB = encodeResultFloats.length;
        int encS = encodeResultFloats[0].length;
        int encH = encodeResultFloats[0][0].length;
        // Flatten encoder hidden states (row-major: [batch, seq, hidden]) using System.arraycopy
        // per row, which avoids the slow element-by-element triple loop and reduces allocation.
        float[] encFlat = new float[encB * encS * encH];
        int encIdx = 0;
        for (int i = 0; i < encB; i++) {
            for (int j = 0; j < encS; j++) {
                System.arraycopy(encodeResultFloats[i][j], 0, encFlat, encIdx, encH);
                encIdx += encH;
            }
        }
        FloatBuffer dataBuffer = FloatBuffer.wrap(encFlat);
        long[] shape = new long[] {encB, encS, encH};
        OnnxTensor encoderHiddenStates = matManager.createOnnxTensor(env, dataBuffer, shape);
        try {
            int batchSize = encB;

            long[][] inputIds = new long[batchSize][maxLength + 1];
            for (int j = 0; j < batchSize; j++) {
                inputIds[j][0] = decoderStartTokenId;
            }

            long curLen = 1;
            boolean[] finished = new boolean[batchSize];
            for (int step = 0; step < maxLength; step++) {
                LongBuffer buffer = LongBuffer.wrap(flattenInputIds(inputIds, (int) curLen));
                OnnxTensor inputIdsTensor = null;
                OrtSession.Result onnxResult = null;
                try {
                    inputIdsTensor = matManager.createOnnxTensor(env, buffer, new long[] {batchSize, curLen});
                    onnxResult = matManager.runSession(session, Map.of(
                            "input_ids", inputIdsTensor,
                            "encoder_hidden_states", encoderHiddenStates));
                    OnnxTensor outputTensor = (OnnxTensor) onnxResult.get(0);
                    long[] outputShape = outputTensor.getInfo().getShape();
                    if (outputShape.length != 3) {
                        throw new FluxException("Unexpected Pix2Text decoder output shape length: " + outputShape.length);
                    }
                    int outputBatchSize = Math.toIntExact(outputShape[0]);
                    int outputSequenceLength = Math.toIntExact(outputShape[1]);
                    int vocabSize = Math.toIntExact(outputShape[2]);
                    if (outputBatchSize != batchSize) {
                        throw new FluxException("Unexpected Pix2Text decoder output batch size: " + outputBatchSize);
                    }
                    FloatBuffer logitsBuffer = outputTensor.getFloatBuffer();
                    for (int j = 0; j < batchSize; j++) {
                        long nextToken;
                        if (finished[j]) {
                            nextToken = padTokenId;
                        } else {
                            nextToken = argmaxLastLogits(logitsBuffer, j, outputSequenceLength, vocabSize);
                            if (nextToken == eosTokenId) {
                                finished[j] = true;
                            }
                        }
                        inputIds[j][(int) curLen] = nextToken;
                    }
                    curLen++;

                    if (ArrayUtil.allTrue(finished)) {
                        break;
                    }
                } finally {
                    matManager.release(onnxResult);
                    matManager.release(inputIdsTensor);
                }
            }

            // Reconstruct generated tokens (everything except the start token, columns 1..curLen-1)
            int outLen = (int) (curLen - 1);
            long[][] generated_tokens = new long[batchSize][];
            for (int j = 0; j < batchSize; j++) {
                long[] tokens = new long[outLen];
                System.arraycopy(inputIds[j], 1, tokens, 0, outLen);
                generated_tokens[j] = tokens;
            }

            List<TextResult> results = new ArrayList<>();
            for (long[] tokens : generated_tokens) {
                String text = tokenizer.decode(tokens, true);
                results.add(new TextResult(text, tokens, -1));
            }
            return results;
        } finally {
            matManager.release(encoderHiddenStates);
        }
    }

    static long[] flattenInputIds(long[][] inputIds, int cols) {
        int rows = inputIds.length;
        long[] flat = new long[rows * cols];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(inputIds[row], 0, flat, row * cols, cols);
        }
        return flat;
    }

    static long argmaxLastLogits(FloatBuffer logitsBuffer, int batchIndex, int sequenceLength, int vocabSize) {
        int offset = ((batchIndex * sequenceLength) + (sequenceLength - 1)) * vocabSize;
        FloatBuffer row = logitsBuffer.duplicate();
        row.position(offset);

        int bestToken = 0;
        float bestScore = row.get();
        for (int token = 1; token < vocabSize; token++) {
            float score = row.get();
            if (score > bestScore) {
                bestScore = score;
                bestToken = token;
            }
        }
        return bestToken;
    }

}
