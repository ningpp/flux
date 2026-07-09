package io.github.flux.blip;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.OnnxSessionUtil;

import java.nio.FloatBuffer;
import java.util.Map;

public class BlipVisionEncoder implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public BlipVisionEncoder(String modelPath, int gpuIndex, OrtEnvironment env) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelPath, gpuIndex);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * @param pixelValues Flattened [batch, 3, 384, 384]
     * @param batchSize   Number of images in batch
     * @return [batch, sequence_length, hidden_size] - image embeddings
     */
    public float[][][] predict(float[] pixelValues, long batchSize) throws OrtException {
        long[] shape = new long[]{batchSize, 3, 384, 384};
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pixelValues), shape);
             Result result = session.run(Map.of("pixel_values", tensor))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
