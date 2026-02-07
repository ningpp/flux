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
        FloatBuffer dataBuffer = FloatBuffer.wrap(ArrayUtil.flat(encodeResultFloats));
        long[] shape = new long[] {
                encodeResultFloats.length,
                encodeResultFloats[0].length,
                encodeResultFloats[0][0].length
        };
        OnnxTensor encoderHiddenStates = OnnxTensor.createTensor(env, dataBuffer, shape);
        int batchSize = (int) shape[0];
        long[][] inputIds = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            inputIds[i] = new long[]{decoderStartTokenId};
        }

        long[][] generated_tokens = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            generated_tokens[i] = new long[0];
        }
        long curLen = 1;
        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < maxLength; i++) {
            long[] flat = ArrayUtil.flat(inputIds);
            LongBuffer buffer = LongBuffer.wrap(flat);
            try (OnnxTensor input_ids_tensor = OnnxTensor.createTensor(env, buffer, new long[] {inputIds.length, inputIds[0].length});
                 OrtSession.Result onnxResult = session.run(Map.of("input_ids", input_ids_tensor, "encoder_hidden_states", encoderHiddenStates))) {
                OnnxValue onnxValue = onnxResult.get(0);
                float[][][] decoderResultFloats = (float[][][]) onnxValue.getValue();
                long[][] nextIds = new long[batchSize][1];
                for (int j = 0; j < batchSize; j++) {
                    if (finished[j]) {
                        nextIds[j][0] = padTokenId;
                        continue;
                    }
                    float[] lastLogit = decoderResultFloats[j][(int) (curLen - 1)];
                    long nextToken = ArrayUtil.argmax(lastLogit);
                    nextIds[j][0] = nextToken;

                    generated_tokens[j] = ArrayUtil.concat(generated_tokens[j], new long[] {nextToken});
                    if (nextToken == eosTokenId) {
                        finished[j] = true;
                    }
                }
                inputIds = ArrayUtil.concat(inputIds, nextIds);
                curLen++;

                if (ArrayUtil.allTrue(finished)) {
                    break;
                }
            }
        }

        IOUtil.close(encoderHiddenStates);

        List<TextResult> results = new ArrayList<>();
        for (long[] tokens : generated_tokens) {
            String text = tokenizer.decode(tokens, true);
            results.add(new TextResult(text, tokens, -1));
        }
        return results;
    }

}
