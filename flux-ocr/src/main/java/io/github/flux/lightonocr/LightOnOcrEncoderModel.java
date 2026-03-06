package io.github.flux.lightonocr;

import ai.djl.ndarray.NDArray;
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
        if (pixelValues == null || pixelValues.isEmpty()) {
            throw new FluxException("pixelValues must not be empty");
        }

        long[] firstShape = pixelValues.get(0).getShape().getShape();
        if (firstShape.length != 3) {
            throw new FluxException("pixel value shape must be [3,H,W], got: " + pixelValues.get(0).getShape());
        }

        int channels = (int) firstShape[0];
        int height = (int) firstShape[1];
        int width = (int) firstShape[2];
        int perImageSize = channels * height * width;

        float[] batchedInput = new float[pixelValues.size() * perImageSize];
        int offset = 0;
        for (NDArray ndArray : pixelValues) {
            long[] shape = ndArray.getShape().getShape();
            if (shape.length != 3 || shape[0] != channels || shape[1] != height || shape[2] != width) {
                throw new FluxException("all pixel values must share shape [" + channels + "," + height
                        + "," + width + "]");
            }

            float[] imageData = ndArray.toFloatArray();
            if (imageData.length != perImageSize) {
                throw new FluxException("invalid flattened pixel size: " + imageData.length);
            }
            System.arraycopy(imageData, 0, batchedInput, offset, perImageSize);
            offset += perImageSize;
        }

        long[] shape = new long[]{pixelValues.size(), channels, height, width};
        FloatBuffer dataBuffer = FloatBuffer.wrap(batchedInput);
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
