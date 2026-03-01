package io.github.flux.gotocr2;

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
import java.util.HashMap;
import java.util.Map;

public class GotOcr2DecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int maxLength;
    private final int layer = 24;
    private final int numHeads = 16;
    private final int headDim = 64;

    public GotOcr2DecoderModel(final String modelFile,
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

    public long[][] predict(float[][][] image_features, long[][] inputIds, float[][][] inputs_embeds,
                              long[][] attentionMask, long[][] positionIds,
                              GotOcr2EmbedModel embedModel) throws OrtException {
        int batch_size = image_features.length;
        Map<String, OnnxTensor> lm_prefill_inputs = new HashMap<>(Map.of(
                "inputs_embeds", ArrayUtil.createOnnxTensor(inputs_embeds, env),
                "attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env),
                "position_ids", ArrayUtil.createOnnxTensor(positionIds, env))
        );
        for (int i = 0; i < layer; i++) {
            OnnxTensor emptyPastKey = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batch_size, numHeads, 0, headDim}
            );
            OnnxTensor emptyPastValue = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(new float[0]),
                    new long[]{batch_size, numHeads, 0, headDim}
            );
            String keyName = "past_key_" + i;
            String valueName = "past_value_" + i;
            lm_prefill_inputs.put(keyName, emptyPastKey);
            lm_prefill_inputs.put(valueName, emptyPastValue);
        }

        Result prefill_result;
        try {
            prefill_result = session.run(lm_prefill_inputs);
        } finally {
            OnnxUtil.closeTensors(lm_prefill_inputs);
        }

        float[][][] logits = (float[][][]) prefill_result.get(0).getValue();
        OnnxTensor[] pkvTensors = new OnnxTensor[layer * 2];
        for (int i = 0; i < layer * 2; i++) {
            pkvTensors[i] = (OnnxTensor) prefill_result.get(i + 1);
        }
        Result kvOwnerResult = prefill_result;
        long stop_id = 151645;

        long start = ArrayUtil.argmax(logits[0][logits[0].length - 1]);
        long[][] generated_tokens = new long[batch_size][];
        for (int i = 0; i < batch_size; i++) {
            generated_tokens[i] = new long[] {start};
        }

        long[][] next_token_ids = new long[batch_size][];
        for (int i = 0; i < batch_size; i++) {
            next_token_ids[i] = new long[] {start};
        }

        int curr_len = inputIds[0].length;
        boolean[] finished = new boolean[batch_size];
        for (int i = 0; i < maxLength; i++) {
            curr_len += 1;

            try (GotOcr2EmbedModel.PredictTensorResult nextEmbedResult = embedModel.predictTensor(next_token_ids)) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("inputs_embeds", nextEmbedResult.embeddings());
                OnnxTensor attentionMaskTensor = ArrayUtil.createOnnxTensor(ArrayUtil.ones(batch_size, curr_len), env);
                inputs.put("attention_mask", attentionMaskTensor);
                long[][] position_ids = new long[batch_size][1];
                for (int j = 0; j < batch_size; j++) {
                    position_ids[j] = new long[]{curr_len - 1};
                }
                OnnxTensor positionIdsTensor = ArrayUtil.createOnnxTensor(position_ids, env);
                inputs.put("position_ids", positionIdsTensor);
                for (int j = 0; j < layer; j++) {
                    inputs.put("past_key_" + j, pkvTensors[2 * j]);
                    inputs.put("past_value_" + j, pkvTensors[2 * j + 1]);
                }

                Result step_out = null;
                try {
                    step_out = session.run(inputs);
                    logits = (float[][][]) step_out.get(0).getValue();
                    OnnxTensor[] nextPkvTensors = new OnnxTensor[layer * 2];
                    for (int o = 0; o < layer * 2; o++) {
                        nextPkvTensors[o] = (OnnxTensor) step_out.get(o + 1);
                    }
                    IOUtil.close(kvOwnerResult);
                    kvOwnerResult = step_out;
                    step_out = null;
                    pkvTensors = nextPkvTensors;
                } finally {
                    IOUtil.close(step_out);
                    IOUtil.close(attentionMaskTensor);
                    IOUtil.close(positionIdsTensor);
                }
            }

            long[][] nextIds = new long[batch_size][1];
            for (int j = 0; j < batch_size; j++) {
                if (finished[j]) {
                    continue;
                }
                float[] lastLogit = logits[j][0];
                long nextToken = ArrayUtil.argmax(lastLogit);
                nextIds[j][0] = nextToken;

                generated_tokens[j] = ArrayUtil.concat(generated_tokens[j], new long[] {nextToken});
                if (nextToken == stop_id) {
                    finished[j] = true;
                }
            }
            next_token_ids = nextIds;

            if (ArrayUtil.allTrue(finished)) {
                break;
            }
        }
        IOUtil.close(kvOwnerResult);
        return generated_tokens;
    }

}
