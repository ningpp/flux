package io.github.flux.llava;

import io.github.flux.core.MatManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Image preprocessing for LLaVA-OneVision with SigLIP vision encoder.
 *
 * Pipeline:
 *   1. Resize to fixed 384x384 (SigLIP image_size)
 *   2. Rescale ÷255
 *   3. Normalize with mean=0.5 std=0.5
 *   4. Return CHW format [3, 384, 384]
 *
 * Unlike Qwen3-VL, this uses a fixed image size without constrained resize or temporal duplication.
 */
public class LlavaOneVisionImageProcessor {

    private static final int IMAGE_SIZE = 384;
    private static final float RESCALE_FACTOR = 1.0f / 255.0f;
    private static final float IMAGE_MEAN = 0.5f;
    private static final float IMAGE_STD = 0.5f;

    /**
     * Result of image preprocessing, ready for the vision encoder.
     * @param pixelValues [3, 384, 384] CHW normalized pixel array
     */
    public record ImageProcessResult(float[][] pixelValues) {}

    /**
     * Process an RGB Mat into pixel_values for the vision encoder.
     *
     * @param rgbMat Input RGB image (any size)
     * @param matManager Mat manager for OpenCV operations
     * @return ImageProcessResult with CHW normalized pixels [3, 384, 384]
     */
    public static ImageProcessResult process(Mat rgbMat, MatManager matManager) {
        int origH = rgbMat.rows();
        int origW = rgbMat.cols();

        // Resize to fixed 384x384
        Mat resized = matManager.newMat();
        Imgproc.resize(rgbMat, resized, new Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, Imgproc.INTER_CUBIC);

        int C = 3;
        int H = resized.rows();
        int W = resized.cols();

        // Build CHW normalized pixel array
        float[] chw = new float[C * H * W];
        byte[] pixel = new byte[C];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                resized.get(y, x, pixel);
                for (int ch = 0; ch < C; ch++) {
                    float val = (pixel[ch] & 0xFF) * RESCALE_FACTOR;
                    val = (val - IMAGE_MEAN) / IMAGE_STD;
                    chw[ch * H * W + y * W + x] = val;
                }
            }
        }

        // Convert CHW flat array to [3][H*W] format for ONNX
        float[][] pixelValues = new float[C][H * W];
        for (int c = 0; c < C; c++) {
            System.arraycopy(chw, c * H * W, pixelValues[c], 0, H * W);
        }

        return new ImageProcessResult(pixelValues);
    }
}
