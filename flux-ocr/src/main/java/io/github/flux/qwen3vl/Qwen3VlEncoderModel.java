package io.github.flux.qwen3vl;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Map;

/**
 * Vision encoder for Qwen3-VL.
 * Uses vision_encoder.onnx which includes ViT + PatchMerger + DeepStack merger.
 *
 * Input:  pixel_values [num_patches, 1536] float32,
 *         image_grid_thw [num_images, 3] int64
 * Output: image_features [num_merged, hidden_size] float32,
 *         deepstack_features_0..N-1 [num_merged, hidden_size] float32
 *
 * The number of deepstack outputs is detected dynamically from the ONNX model
 * (total outputs - 1 for image_features).
 */
public class Qwen3VlEncoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int numDeepstack;

    public Qwen3VlEncoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
            // Detect deepstack count from ONNX model outputs: total outputs - 1 (image_features)
            this.numDeepstack = (int) (session.getNumOutputs() - 1);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Result containing image features and N deepstack feature tensors (all [num_merged, hidden_size]).
     * @param imageFeatures  [num_merged, hidden_size]
     * @param deepstackFeatures [N][num_merged][hidden_size] where N = numDeepstack
     */
    public record EncoderResult(
            float[][] imageFeatures,
            float[][][] deepstackFeatures
    ) {}

    /**
     * Run vision encoder inference.
     *
     * @param pixelValues   [num_patches, 1536] from image processor
     * @param imageGridThw  [num_images, 3] grid dimensions
     * @return EncoderResult with image features and deepstack tensors
     */
    public EncoderResult predict(float[][] pixelValues, long[][] imageGridThw) throws OrtException {
        int numPatches = pixelValues.length;
        int patchDim = pixelValues[0].length;

        // Flatten pixel_values to FloatBuffer
        FloatBuffer pvBuffer = FloatBuffer.allocate(numPatches * patchDim);
        for (float[] patch : pixelValues) {
            pvBuffer.put(patch);
        }
        pvBuffer.flip();

        int numImages = imageGridThw.length;
        LongBuffer gridBuffer = LongBuffer.allocate(numImages * 3);
        for (long[] grid : imageGridThw) {
            gridBuffer.put(grid);
        }
        gridBuffer.flip();

        try (OnnxTensor pvTensor = OnnxTensor.createTensor(env, pvBuffer,
                     new long[]{numPatches, patchDim});
             OnnxTensor gridTensor = OnnxTensor.createTensor(env, gridBuffer,
                     new long[]{numImages, 3});
             Result result = session.run(Map.of(
                     "pixel_values", pvTensor,
                     "image_grid_thw", gridTensor))) {

            float[][] imageFeatures = (float[][]) result.get(0).getValue();
            float[][][] deepstackFeatures = new float[numDeepstack][][];
            for (int i = 0; i < numDeepstack; i++) {
                deepstackFeatures[i] = (float[][]) result.get(i + 1).getValue();
            }

            return new EncoderResult(imageFeatures, deepstackFeatures);
        }
    }

    /**
     * @return the number of deepstack feature levels detected from the ONNX model.
     */
    public int getNumDeepstack() {
        return numDeepstack;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
