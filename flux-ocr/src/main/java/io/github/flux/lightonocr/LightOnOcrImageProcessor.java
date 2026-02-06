package io.github.flux.lightonocr;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.MatManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Image preprocessing for LightOnOCR-2-1B (Pixtral-style).
 * Resize longest edge to 1540, round up to multiples of patch_size (14),
 * rescale ÷255, CLIP normalize, transpose to CHW.
 */
public class LightOnOcrImageProcessor {

    private static final int PATCH_SIZE = 14;
    private static final int SPATIAL_MERGE_SIZE = 2;
    private static final int EFFECTIVE_PATCH_SIZE = PATCH_SIZE * SPATIAL_MERGE_SIZE; // 28
    private static final int LONGEST_EDGE = 1540;

    private static final float[] IMAGE_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] IMAGE_STD  = {0.26862954f, 0.26130258f, 0.27577711f};
    private static final float RESCALE_FACTOR = 1.0f / 255.0f;

    public record SizeConfig(int height, int width, int numRows, int numCols) {
        public int totalImageTokens() {
            return numRows * numCols;
        }
    }

    /**
     * Compute target size and token grid dimensions for an image.
     */
    public static SizeConfig computeSizeConfig(int origHeight, int origWidth) {
        int h = origHeight;
        int w = origWidth;

        // Scale down so longest edge <= LONGEST_EDGE
        double ratio = Math.max((double) h / LONGEST_EDGE, (double) w / LONGEST_EDGE);
        if (ratio > 1.0) {
            h = (int) Math.floor(h / ratio);
            w = (int) Math.floor(w / ratio);
        }

        // Round UP to nearest multiple of EFFECTIVE_PATCH_SIZE (28)
        // ONNX vision encoder's patch merger requires even patch grid dimensions
        int newH = ((h - 1) / EFFECTIVE_PATCH_SIZE + 1) * EFFECTIVE_PATCH_SIZE;
        int newW = ((w - 1) / EFFECTIVE_PATCH_SIZE + 1) * EFFECTIVE_PATCH_SIZE;

        // Token grid (floor division by effective patch size 28)
        int numRows = newH / EFFECTIVE_PATCH_SIZE;
        int numCols = newW / EFFECTIVE_PATCH_SIZE;

        return new SizeConfig(newH, newW, numRows, numCols);
    }

    /**
     * Preprocess an RGB Mat image for the vision encoder.
     *
     * @return NDArray of shape [3, H, W] (CHW, float32, normalized)
     */
    public static NDArray process(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        int origH = rgbMat.rows();
        int origW = rgbMat.cols();
        SizeConfig sizeConfig = computeSizeConfig(origH, origW);

        // Resize with bicubic interpolation
        Mat resized = matManager.newMat();
        Imgproc.resize(rgbMat, resized, new Size(sizeConfig.width(), sizeConfig.height()), 0, 0, Imgproc.INTER_CUBIC);

        int h = resized.rows();
        int w = resized.cols();
        int c = resized.channels(); // 3

        // Convert to float CHW array with rescale and normalize
        float[] chw = new float[c * h * w];
        byte[] pixel = new byte[c];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                resized.get(y, x, pixel);
                for (int ch = 0; ch < c; ch++) {
                    float val = (pixel[ch] & 0xFF) * RESCALE_FACTOR;
                    val = (val - IMAGE_MEAN[ch]) / IMAGE_STD[ch];
                    chw[ch * h * w + y * w + x] = val;
                }
            }
        }

        return ndManager.create(chw, new ai.djl.ndarray.types.Shape(c, h, w));
    }
}
