package io.github.flux.gotocr2;

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
                              GotOcr2EmbedModel embedModel,
                              NDManager ndManager) throws OrtException {
        int batch_size = image_features.length;
        int layer = 24;
        int num_heads = 16;
        int head_dim = 64;
        Map<String, OnnxTensor> lm_prefill_inputs = new HashMap<>(Map.of(
            "inputs_embeds", ArrayUtil.createOnnxTensor(inputs_embeds, env),
            "attention_mask", ArrayUtil.createOnnxTensor(attentionMask, env),
            "position_ids", ArrayUtil.createOnnxTensor(positionIds, env))
        );
        for (int i = 0; i < layer; i++) {
            NDArray past_key_value = ndManager.zeros(new Shape(batch_size, num_heads, 0, head_dim), DataType.FLOAT32);
            long[] past_key_value_shape = past_key_value.getShape().getShape();
            FloatBuffer past_key_value_buffer = past_key_value.toByteBuffer().asFloatBuffer();
            lm_prefill_inputs.put("past_key_" + i, OnnxTensor.createTensor(env, past_key_value_buffer, past_key_value_shape));
            lm_prefill_inputs.put("past_value_" + i, OnnxTensor.createTensor(env, past_key_value_buffer, past_key_value_shape));
        }

        Result prefill_result = session.run(lm_prefill_inputs);
        float[][][] logits = (float[][][]) prefill_result.get(0).getValue();
        float[][][][][] pkvs = new float[layer*2][][][][];
        for (int i = 0; i < layer*2; i++) {
            pkvs[i] = (float[][][][]) prefill_result.get(i+1).getValue();
        }
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
            float[][][] next_embed = embedModel.predict(next_token_ids);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(next_embed, env));
            inputs.put("attention_mask", ArrayUtil.createOnnxTensor(ArrayUtil.ones(batch_size, curr_len), env));
            long[][] position_ids = new long[batch_size][1];
            for (int j = 0; j < batch_size; j++) {
                position_ids[j] = new long[] {curr_len-1};
            }
            inputs.put("position_ids", ArrayUtil.createOnnxTensor( position_ids, env));
            for (int j = 0; j < layer; j++) {
                inputs.put("past_key_" + j, ArrayUtil.createOnnxTensor(pkvs[2*j], env));
                inputs.put("past_value_" + j, ArrayUtil.createOnnxTensor(pkvs[2*j+1], env));
            }
            Result step_out = session.run(inputs);
            logits = (float[][][]) step_out.get(0).getValue();
            for (int o = 0; o < layer*2; o++) {
                pkvs[o] = (float[][][][]) step_out.get(o+1).getValue();
            }
            IOUtil.close(step_out);
            OnnxUtil.closeTensors(inputs);

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
        return generated_tokens;
    }

}
