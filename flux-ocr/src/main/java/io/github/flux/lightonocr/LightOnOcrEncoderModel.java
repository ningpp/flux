package io.github.flux.lightonocr;

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

/**
 * Vision encoder for LightOnOCR-2-1B (Pixtral ViT + PatchMerger + MLP projector).
 * Input:  pixel_values [batch, 3, H, W] float32
 * Output: image_features [batch, num_merged_patches, 1024] float32
 */
public class LightOnOcrEncoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public LightOnOcrEncoderModel(final String modelFile,
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
     * Run vision encoder on a batch of preprocessed images.
     *
     * @param pixelValues list of NDArray, each [3, H, W]
     * @return image_features [batch, num_merged_patches, 1024]
     */
    public float[][][] predict(List<NDArray> pixelValues) throws OrtException {
        NDList ndList = new NDList(pixelValues);
        NDArray inputNdArray = NDArrays.stack(ndList);
        long[] shape = inputNdArray.getShape().getShape();
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
             Result result = session.run(Map.of("pixel_values", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
