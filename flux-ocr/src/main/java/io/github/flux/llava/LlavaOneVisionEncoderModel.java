package io.github.flux.llava;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.util.Map;

/**
 * Vision encoder for LLaVA-OneVision-Qwen2.
 * Uses SigLIP vision encoder (vision_encoder.onnx).
 *
 * Input:  pixel_values [batch, 3, 384, 384] float32
 * Output: image_features [batch, 729, hidden_size] float32
 *         where 729 = 27*27 (SigLIP patch grid)
 *
 * Unlike Qwen3-VL, this model has no deepstack features.
 */
public class LlavaOneVisionEncoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public LlavaOneVisionEncoderModel(final String modelFile,
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
     * Run vision encoder inference.
     *
     * @param pixelValues [batch, 3, 384*384] CHW flattened format from image processor
     *                    Each inner array is [H*W] for one channel
     * @return [batch, 729, hidden_size] image features
     */
    public float[][][] predict(float[][][] pixelValues) throws OrtException {
        int batchSize = pixelValues.length;
        int C = pixelValues[0].length;  // 3 channels
        int HxW = pixelValues[0][0].length;  // 384*384

        // Reconstruct CHW flat array: [batch, C*H*W]
        float[] chwBatch = new float[batchSize * C * HxW];
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < C; c++) {
                System.arraycopy(pixelValues[b][c], 0, chwBatch, b * C * HxW + c * HxW, HxW);
            }
        }

        FloatBuffer buffer = FloatBuffer.wrap(chwBatch);

        try (OnnxTensor pvTensor = OnnxTensor.createTensor(env, buffer,
                     new long[]{batchSize, C, 384, 384});
             OrtSession.Result result = session.run(Map.of("pixel_values", pvTensor))) {

            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
