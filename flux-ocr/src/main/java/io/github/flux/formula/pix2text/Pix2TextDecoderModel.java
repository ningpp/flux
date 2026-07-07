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
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;

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
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public List<TextResult> batchPredict(float[][][] encodeResultFloats) throws OrtException {
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
        OnnxTensor encoderHiddenStates = OnnxTensor.createTensor(env, dataBuffer, shape);
        try {
            int batchSize = encB;

            // Performance: avoid reallocating inputIds (ArrayUtil.concat) and re-flattening
            // (ArrayUtil.flat) on every autoregressive step. Use a single preallocated flat
            // buffer of size [batch * (maxLength + 1)] laid out row-major as [batch, seqLen]:
            //   input_ids[j][pos] == flatInput[pos * batchSize + j]
            // This makes each decoding step allocation-free (only a reused LongBuffer view).
            long[] flatInput = new long[batchSize * (maxLength + 1)];
            for (int j = 0; j < batchSize; j++) {
                flatInput[j] = decoderStartTokenId; // column 0
            }

            long[][] generated_tokens = new long[batchSize][];
            for (int j = 0; j < batchSize; j++) {
                generated_tokens[j] = new long[0];
            }
            long curLen = 1;
            boolean[] finished = new boolean[batchSize];
            for (int step = 0; step < maxLength; step++) {
                LongBuffer buffer = LongBuffer.wrap(flatInput).position(0).limit((int) (batchSize * curLen));
                try (OnnxTensor input_ids_tensor = OnnxTensor.createTensor(env, buffer, new long[] {batchSize, curLen});
                     OrtSession.Result onnxResult = session.run(Map.of("input_ids", input_ids_tensor, "encoder_hidden_states", encoderHiddenStates))) {
                    // OnnxValue obtained from Result is owned by Result and will be closed when Result closes.
                    float[][][] decoderResultFloats = (float[][][]) onnxResult.get(0).getValue();
                    for (int j = 0; j < batchSize; j++) {
                        long nextToken;
                        if (finished[j]) {
                            nextToken = padTokenId;
                        } else {
                            float[] lastLogit = decoderResultFloats[j][(int) (curLen - 1)];
                            nextToken = ArrayUtil.argmax(lastLogit);
                            if (nextToken == eosTokenId) {
                                finished[j] = true;
                            }
                        }
                        // write the next token at column `curLen`
                        flatInput[(int) (curLen * batchSize + j)] = nextToken;
                    }
                    curLen++;

                    if (ArrayUtil.allTrue(finished)) {
                        break;
                    }
                }
            }

            // Reconstruct generated tokens (everything except the start token, columns 1..curLen-1)
            int outLen = (int) (curLen - 1);
            for (int j = 0; j < batchSize; j++) {
                long[] tokens = new long[outLen];
                for (int k = 1; k <= outLen; k++) {
                    tokens[k - 1] = flatInput[k * batchSize + j];
                }
                generated_tokens[j] = tokens;
            }

            List<TextResult> results = new ArrayList<>();
            for (long[] tokens : generated_tokens) {
                String text = tokenizer.decode(tokens, true);
                results.add(new TextResult(text, tokens, -1));
            }
            return results;
        } finally {
            IOUtil.close(encoderHiddenStates);
        }
    }

}
