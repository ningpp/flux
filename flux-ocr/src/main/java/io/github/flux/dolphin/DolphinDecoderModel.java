// this code is convert from  https://github.com/bytedance/Dolphin/blob/v1.5
// Dolphin v1.5 IS Licensed under the MIT License
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
package io.github.flux.dolphin;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class DolphinDecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;
    private final int maxLength;
    private final long padTokenId;
    private final long eosTokenId;
    private final HuggingFaceTokenizer tokenizer;

    public DolphinDecoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               int maxLength,
                               long padTokenId,
                               long eosTokenId,
                               HuggingFaceTokenizer tokenizer) {
        this.maxLength = maxLength;
        this.padTokenId = padTokenId;
        this.eosTokenId = eosTokenId;
        this.tokenizer = tokenizer;
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

    public List<TextResult> predict(String prompt, OnnxTensor encoder_hidden_states_tensor, long[] decoder_input_ids, NDManager manager, boolean skipSpecialTokens) throws OrtException {
        int batchSize = Math.toIntExact(encoder_hidden_states_tensor.getInfo().getShape()[0]);
        long[][] inputIds = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            inputIds[i] = ArrayUtil.clone(decoder_input_ids);
        }

        long[][] generated_tokens = new long[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            generated_tokens[i] = new long[0];
        }
        long curLen = decoder_input_ids.length;
        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < maxLength; i++) {

            Map<String, OnnxTensor> inputs = new HashMap<>(2);
            long[] flat = ArrayUtil.flat(inputIds);
            LongBuffer buffer = LongBuffer.wrap(flat);
            OnnxTensor input_ids_tensor = OnnxTensor.createTensor(env, buffer,
                    new long[] {inputIds.length, inputIds[0].length});
            inputs.put("input_ids", input_ids_tensor);
            inputs.put("encoder_hidden_states", encoder_hidden_states_tensor);

            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            if (optinalResult.isPresent()) {
                float[][][] decoderResultFloats = (float[][][]) optinalResult.get().getValue();
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
            IOUtil.close(onnxResult);
            // don't close encoder_hidden_states here
            inputs.put("encoder_hidden_states", null);
            IOUtil.close(input_ids_tensor);
        }

        List<TextResult> results = new ArrayList<>();
        for (long[] tokens : generated_tokens) {
            String text = tokenizer.decode(tokens, skipSpecialTokens);
            String noPromptText = text.replace(prompt, "").replace("<pad>", "").replace("</s>", "").strip();
            results.add(new TextResult(noPromptText, tokens, -1));
        }
        return results;
    }

    private static long[] getNextTokenIds(float[][][] decoderResultFloats) {
        // 获取维度信息
        int batchSize = decoderResultFloats.length;
        int seqLength = decoderResultFloats[0].length;
        int vocabSize = decoderResultFloats[0][0].length;

        // 提取 [:, -1, :] - 取每个batch的最后一个token的logits
        float[][] lastTokenLogits = new float[batchSize][vocabSize];
        for (int i = 0; i < batchSize; i++) {
            lastTokenLogits[i] = decoderResultFloats[i][seqLength - 1];
        }

        // 对最后一维进行 argMax，得到每个batch的下一个token id
        long[] nextTokenIds = new long[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int maxIndex = 0;
            float maxValue = lastTokenLogits[i][0];
            for (int j = 1; j < vocabSize; j++) {
                if (lastTokenLogits[i][j] > maxValue) {
                    maxValue = lastTokenLogits[i][j];
                    maxIndex = j;
                }
            }
            nextTokenIds[i] = maxIndex;
        }
        return nextTokenIds;
    }

}
