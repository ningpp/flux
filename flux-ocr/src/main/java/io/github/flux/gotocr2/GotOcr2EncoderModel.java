package io.github.flux.gotocr2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class GotOcr2EncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public GotOcr2EncoderModel(final String modelFile,
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

    public float[][][] predict(List<NDArray> pixel_values) throws OrtException {
        NDList ndList = new NDList(pixel_values);
        NDArray inputNdArray = NDArrays.stack(ndList);
        long[] shape = inputNdArray.getShape().getShape();
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
             Result result = session.run(Map.of("pixel_values", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

}
