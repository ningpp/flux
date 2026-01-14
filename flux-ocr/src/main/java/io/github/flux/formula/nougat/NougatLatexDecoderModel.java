// this code is convert from https://github.com/NormXU/nougat-latex-ocr
// nougat-latex-ocr's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.nougat;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.FormulaRecognitionResult;
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
import java.util.Optional;
import java.util.Set;

public class NougatLatexDecoderModel implements AutoCloseable {

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
    private final long decoderStartTokenId;
    private final HuggingFaceTokenizer tokenizer;

    public NougatLatexDecoderModel(final String modelFile,
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
        this.decoderStartTokenId = decoderStartTokenId;
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

    public List<FormulaRecognitionResult> batchPredict(float[][][] encodeResultFloats, NDManager manager) throws OrtException {
        NDArray inputNdArray = ArrayUtil.toNDArray(manager, encodeResultFloats);
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        long[] shape = inputNdArray.getShape().getShape();
        OnnxTensor encoder_hidden_states_tensor = OnnxTensor.createTensor(env, dataBuffer, shape);

        int batchSize = encodeResultFloats.length;
        long[][] inputIds = new long[batchSize][];
        for (int i = 0; i < encodeResultFloats.length; i++) {
            inputIds[i] = ArrayUtil.clone(new long[]{decoderStartTokenId});
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
            OnnxTensor input_ids_tensor = OnnxTensor.createTensor(env, buffer,
                    new long[] {inputIds.length, inputIds[0].length});
            Map<String, OnnxTensor> inputs = new HashMap<>(2);
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
            IOUtil.close(inputNdArray);
            IOUtil.close(input_ids_tensor);
        }

        List<FormulaRecognitionResult> results = new ArrayList<>();
        for (long[] tokens : generated_tokens) {
            String text = tokenizer.decode(tokens, true);
            results.add(new FormulaRecognitionResult(List.of(text), tokens, -1));
        }
        return results;
    }

    public FormulaRecognitionResult predict(float[][][] encodeResultFloats, NDManager manager) throws OrtException {
        NDArray inputNdArray = ArrayUtil.toNDArray(manager, encodeResultFloats);
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        long[] shape = inputNdArray.getShape().getShape();
        OnnxTensor encoder_hidden_states_tensor = OnnxTensor.createTensor(env, dataBuffer, shape);

        long[] inputIds = new long[]{decoderStartTokenId};
        for (int i = 0; i < maxLength; i++) {

            OnnxTensor input_ids_tensor = OnnxTensor.createTensor(env, new long[][]{inputIds});
            Map<String, OnnxTensor> inputs = new HashMap<>(2);
            inputs.put("input_ids", input_ids_tensor);
            inputs.put("encoder_hidden_states", encoder_hidden_states_tensor);
            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            if (optinalResult.isPresent()) {
                float[][][] decoderResultFloats = (float[][][]) optinalResult.get().getValue();
                NDArray decoderResults = ArrayUtil.toNDArray(manager, decoderResultFloats);
                // 等价于 [:, -1, :]
                NDArray lastTokenLogits = decoderResults.get(":, -1, :");

                NDArray nextTokenLogits = lastTokenLogits.toType(DataType.FLOAT32, true);

                NDArray _nextTokens = nextTokenLogits.argMax(-1);
                NDArray nextTokens = _nextTokens.toType(DataType.INT64, false);

                long[] nextTokenIds = nextTokens.toLongArray();
                inputIds = ArrayUtil.concat(inputIds, nextTokenIds);
                if (ArrayUtil.contains(nextTokenIds, eosTokenId)) {
                    break;
                }
            }
            IOUtil.close(onnxResult);
            IOUtil.close(inputNdArray);
            IOUtil.close(input_ids_tensor);
        }

        String text = tokenizer.decode(inputIds, true);
        NDArray _sequences = manager.create(inputIds);
        NDArray sequences = _sequences.reshape(1, inputIds.length);
        var r = new FormulaRecognitionResult(List.of(text), inputIds, -1);
        sequences.close();
        _sequences.close();
        return r;
    }

    public FormulaRecognitionResult predict(float[][][] encodeResultFloats, long[] decoder_input_ids, NDManager manager) throws OrtException {
        long[] inputIds = ArrayUtil.clone(decoder_input_ids);
        List<float[]> scores = new ArrayList<>();

        for (int i = 0; i < maxLength; i++) {
            int d1 = encodeResultFloats.length;
            int d2 = encodeResultFloats[0].length;
            int d3 = encodeResultFloats[0][0].length;
            long[] shape = new long[]{d1, d2, d3};
            FloatBuffer dataBuffer = FloatBuffer.wrap(ArrayUtil.flat(encodeResultFloats));
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>(2);
            inputs.put("input_ids", OnnxTensor.createTensor(env, new long[][]{inputIds}));
            inputs.put("encoder_hidden_states", onnxInput);

            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            if (optinalResult.isPresent()) {
                float[][][] decoderResultFloats = (float[][][]) optinalResult.get().getValue();
                NDArray decoderResults = ArrayUtil.toNDArray(manager, decoderResultFloats);
                // 等价于 [:, -1, :]
                NDArray lastTokenLogits = decoderResults.get(":, -1, :");

                NDArray nextTokenLogits = lastTokenLogits.toType(DataType.FLOAT32, true);
                scores.add(nextTokenLogits.toFloatArray());

                NDArray _nextTokens = nextTokenLogits.argMax(-1);
                NDArray nextTokens = _nextTokens.toType(DataType.INT64, false);

                long[] nextTokenIds = nextTokens.toLongArray();

                nextTokens.close();
                _nextTokens.close();
                lastTokenLogits.close();
                decoderResults.close();
                nextTokenLogits.close();

                inputIds = ArrayUtil.concat(inputIds, nextTokenIds);
                if (ArrayUtil.contains(nextTokenIds, eosTokenId)) {
                    break;
                }
            }
            onnxResult.close();
            OnnxUtil.closeTensors(inputs);
        }

        String text = tokenizer.decode(inputIds, true);
        NDArray _sequences = manager.create(inputIds);
        NDArray sequences = _sequences.reshape(1, inputIds.length);

        NDList scoresNdList = new NDList();
        for (float[] flat : scores) {
            scoresNdList.add(manager.create(flat, new Shape(1, flat.length)));
        }
        var result = new FormulaRecognitionResult(List.of(text), inputIds, -1);
        sequences.close();
        _sequences.close();
        scoresNdList.close();
        return result;
    }
}
