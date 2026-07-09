package io.github.flux.gotocr2;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.util.OnnxUtil;

import java.util.Map;

public class GotOcr2EmbedModel implements AutoCloseable {

    public record PredictTensorResult(OnnxTensor embeddings, Result onnxResult) implements AutoCloseable {

        @Override
        public void close() {
            try {
                onnxResult.close();
            } catch (Exception ignore) {
            }
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public GotOcr2EmbedModel(final String modelFile,
                             final int gpuIndex,
                             final OrtEnvironment env) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
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

    public PredictTensorResult predictTensor(long[][] inputIds) throws OrtException {
        OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
        Map<String, OnnxTensor> inputs = Map.of("input_ids", onnxInput);
        Result result = null;
        try {
            result = session.run(inputs);
            OnnxTensor embeddings = (OnnxTensor) result.get(0);
            onnxInput.close();
            return new PredictTensorResult(embeddings, result);
        } catch (Exception e) {
            OnnxUtil.closeTensors(inputs);
            if (result != null) {
                result.close();
            }
            throw e;
        }
    }

}
