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
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class GotOcr2DecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;

    public GotOcr2DecoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env) {
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

    public List<Long> predict(float[][][] image_features, long[][] inputIds, float[][][] inputs_embeds,
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
            float[][][][] empty_key = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
            float[][][][] empty_value = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
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

        long next_token_id = ArrayUtil.argmax(logits[0][logits[0].length - 1]);
        List<Long> generated_ids = new ArrayList<>();
        generated_ids.add(next_token_id);

        int curr_len = inputIds[0].length;
        for (int i = 0; i < 32; i++) {
            System.out.println(String.format("%6d", i) + " ".repeat(11) + LocalDateTime.now());
            float[][][] next_embed = embedModel.predict(new long[][] { {next_token_id} });

            curr_len += 1;
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("inputs_embeds", ArrayUtil.createOnnxTensor(next_embed, env));
            inputs.put("attention_mask", ArrayUtil.createOnnxTensor(ones(1, curr_len), env));
            inputs.put("position_ids", ArrayUtil.createOnnxTensor( new long[][] { { (long)curr_len - 1L } }, env));
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
            next_token_id = nextId(logits);
            if (next_token_id == stop_id) {
                break;
            }
            generated_ids.add(next_token_id);
        }
        System.out.println("\n".repeat(11));
        System.out.println("generated_ids");
        System.out.println(String.join(", ", generated_ids.stream().map(String::valueOf).toList()));
        return generated_ids;
    }

    public static long[][] ones(int rows, int cols) {
        long[][] a = new long[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) a[i][j] = 1L;
        }
        return a;
    }

    private long nextId(float[][][] logits) {
        float[] last = logits[0][logits[0].length - 1];
        int nextTokenId = 0;
        float max = last[0];
        for (int i = 1; i < last.length; i++) {
            if (last[i] > max) {
                max = last[i];
                nextTokenId = i;
            }
        }
        return nextTokenId;
    }

}
