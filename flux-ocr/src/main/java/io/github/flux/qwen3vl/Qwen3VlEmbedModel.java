package io.github.flux.qwen3vl;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.Map;

/**
 * Token embedding for Qwen3-VL-2B-Instruct (embed_tokens, tied with lm_head).
 * Input:  input_ids [batch, seq_len] int64
 * Output: inputs_embeds [batch, seq_len, 2048] float32
 */
public class Qwen3VlEmbedModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public Qwen3VlEmbedModel(final String modelFile,
                             final int gpuIndex,
                             final OrtEnvironment env) {
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

    public float[][][] predict(long[][] inputIds) throws OrtException {
        try (OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
             Result result = session.run(Map.of("input_ids", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
