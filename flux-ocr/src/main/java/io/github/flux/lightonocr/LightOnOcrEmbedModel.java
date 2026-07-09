package io.github.flux.lightonocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxSessionUtil;

import java.util.Map;

/**
 * Token embedding for LightOnOCR-2-1B (Qwen3 embed_tokens, tied with lm_head).
 * Input:  input_ids [batch, seq_len] int64
 * Output: inputs_embeds [batch, seq_len, 1024] float32
 */
public class LightOnOcrEmbedModel implements AutoCloseable {

    public record EmbedOutput(OnnxTensor tensor, Result result) implements AutoCloseable {
        @Override
        public void close() throws Exception {
            IOUtil.close(result);
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public LightOnOcrEmbedModel(final String modelFile,
                                final int gpuIndex,
                                final OrtEnvironment env) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Embed token IDs into dense vectors.
     *
     * @param inputIds [batch, seq_len]
     * @return inputs_embeds [batch, seq_len, 1024]
     */
    public float[][][] predict(long[][] inputIds) throws OrtException {
        try (OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
             Result result = session.run(Map.of("input_ids", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    /**
     * Embed token IDs and return tensor output directly for zero-copy decode path.
     * Caller must close returned EmbedOutput after the tensor is consumed.
     */
    public EmbedOutput predictTensor(long[][] inputIds) throws OrtException {
        OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
        try {
            Result result = session.run(Map.of("input_ids", onnxInput));
            return new EmbedOutput((OnnxTensor) result.get(0), result);
        } finally {
            if (!onnxInput.isClosed()) {
                onnxInput.close();
            }
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
