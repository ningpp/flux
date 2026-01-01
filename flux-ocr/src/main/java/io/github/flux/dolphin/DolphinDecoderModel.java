package io.github.flux.dolphin;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
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
    private final boolean skipSpecialTokens;
    private final OnnxJavaType dtype;

    public DolphinDecoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               int maxLength,
                               long padTokenId,
                               long eosTokenId,
                               HuggingFaceTokenizer tokenizer,
                               boolean skipSpecialTokens,
                               final OnnxJavaType dtype) {
        this.dtype = dtype;
        this.maxLength = maxLength;
        this.padTokenId = padTokenId;
        this.eosTokenId = eosTokenId;
        this.tokenizer = tokenizer;
        this.skipSpecialTokens = skipSpecialTokens;
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

    public List<DolphinElementResult> predict(String prompt, float[][][] encodeResultFloats, long[] decoder_input_ids, NDManager manager) throws OrtException {
        int batchSize = encodeResultFloats.length;
        long[][] inputIds = new long[batchSize][];
        for (int i = 0; i < encodeResultFloats.length; i++) {
            inputIds[i] = ArrayUtil.clone(decoder_input_ids);
        }
        int d1 = encodeResultFloats.length;
        int d2 = encodeResultFloats[0].length;
        int d3 = encodeResultFloats[0][0].length;
        long[] shape = new long[]{d1, d2, d3};
        OnnxTensor encoder_hidden_states_tensor;
        if (dtype == OnnxJavaType.FLOAT) {
            FloatBuffer dataBuffer = FloatBuffer.wrap(ArrayUtil.flat(encodeResultFloats));
            encoder_hidden_states_tensor = OnnxTensor.createTensor(env, dataBuffer, shape);
        } else {
            NDArray encoded = ArrayUtil.toNDArray(manager, encodeResultFloats);
            NDArray encodedFloat16 = encoded.toType(DataType.FLOAT16, true);
            ShortBuffer buffer = encodedFloat16.toByteBuffer().asShortBuffer();
            encoder_hidden_states_tensor = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16);
            IOUtil.close(encodedFloat16);
            IOUtil.close(encoded);
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

        IOUtil.close(encoder_hidden_states_tensor);
        List<DolphinElementResult> results = new ArrayList<>();
        for (long[] tokens : generated_tokens) {
            String text = tokenizer.decode(tokens, skipSpecialTokens);
            String noPromptText = text.replace(prompt, "").replace("<pad>", "").replace("</s>", "").strip();
            results.add(new DolphinElementResult(noPromptText, tokens, -1));
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
