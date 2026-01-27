package io.github.flux.gotocr2;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.HashMap;
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

    public void predict(float[][][] image_features, float[][][] inputs_embeds,
                        long[][] attentionMask, long[][] positionIds) throws OrtException {
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
            lm_prefill_inputs.put("past_key_" + i, ArrayUtil.createOnnxTensor(empty_key, env));
            lm_prefill_inputs.put("past_value_" + i, ArrayUtil.createOnnxTensor(empty_key, env));
        }

        Result prefill_result = session.run(lm_prefill_inputs);
        float[][][] logits = (float[][][]) prefill_result.get(0).getValue();
        float[][][][][] pkvs = new float[layer][][][][];
        for (int i = 0; i < layer; i++) {
            pkvs[i] = (float[][][][]) prefill_result.get(i+1).getValue();
        }
        long eos_token_id = 151643;
    }

}
