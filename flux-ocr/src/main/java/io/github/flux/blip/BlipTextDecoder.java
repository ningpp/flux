package io.github.flux.blip;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.Map;

public class BlipTextDecoder implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public BlipTextDecoder(String modelPath, int gpuIndex, OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelPath, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] predict(long[][] inputIds, float[][][] encoderHiddenStates) throws OrtException {
        try (OnnxTensor idsTensor = ArrayUtil.createOnnxTensor(inputIds, env);
             OnnxTensor attTensor = ArrayUtil.createOnnxTensor(encoderHiddenStates, env);
             Result result = session.run(Map.of(
                     "input_ids", idsTensor,
                     "encoder_hidden_states", attTensor
             ))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
