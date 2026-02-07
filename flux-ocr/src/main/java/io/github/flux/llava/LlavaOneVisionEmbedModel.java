package io.github.flux.llava;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.Map;

/**
 * Token embedding for LLaVA-OneVision-Qwen2 (embed_tokens, tied with lm_head).
 * Input:  input_ids [batch, seq_len] int64
 * Output: inputs_embeds [batch, seq_len, hidden_size] float32 (hidden_size=896)
 */
public class LlavaOneVisionEmbedModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public LlavaOneVisionEmbedModel(final String modelFile,
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

    /**
     * Convert token IDs to embeddings.
     *
     * @param inputIds [batch, seq_len] token IDs
     * @return [batch, seq_len, hidden_size] embeddings
     */
    public float[][][] predict(long[][] inputIds) throws OrtException {
        try (OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
             OrtSession.Result result = session.run(Map.of("input_ids", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
