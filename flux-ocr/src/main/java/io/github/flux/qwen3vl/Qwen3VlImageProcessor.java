package io.github.flux.qwen3vl;

import io.github.flux.core.MatManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Image preprocessing for Qwen3-VL-2B-Instruct.
 * Matches the transformers Qwen2VLImageProcessorFast._preprocess pipeline:
 *   1. smart_resize (factor=32, min=65536, max=16777216 pixels)
 *   2. Rescale ÷255, normalize with mean=0.5 std=0.5
 *   3. Temporal duplication (temporal_patch_size=2)
 *   4. Patch extraction with merge_size=2 grouping
 *
 * Output: pixel_values [num_patches, 1536], image_grid_thw [1, 3].
 */
public class Qwen3VlImageProcessor {

    private static final int PATCH_SIZE = 16;
    private static final int TEMPORAL_PATCH_SIZE = 2;
    private static final int MERGE_SIZE = 2;
    private static final int FACTOR = PATCH_SIZE * MERGE_SIZE;  // 32

    private static final int MIN_PIXELS = 65536;
    private static final int MAX_PIXELS = 16777216;

    private static final float RESCALE_FACTOR = 1.0f / 255.0f;
    private static final float IMAGE_MEAN = 0.5f;
    private static final float IMAGE_STD = 0.5f;

    /**
     * Result of image preprocessing, ready for the vision encoder.
     */
    public record ImageProcessResult(
            float[][] pixelValues,    // [num_patches, 1536]
            long[][] imageGridThw,    // [1, 3] = {{grid_t, grid_h, grid_w}}
            int numMergedTokens       // grid_t * (grid_h / merge) * (grid_w / merge)
    ) {}

    /**
     * Resize dimensions to multiples of factor, respecting pixel count limits.
     */
    public static int[] smartResize(int height, int width) {
        if (height < FACTOR || width < FACTOR) {
            throw new IllegalArgumentException(
                    "Image too small: " + height + "x" + width + ", min factor=" + FACTOR);
        }
        int hBar = (int) (Math.round((double) height / FACTOR) * FACTOR);
        int wBar = (int) (Math.round((double) width / FACTOR) * FACTOR);
        if ((long) hBar * wBar < MIN_PIXELS) {
            double beta = Math.sqrt((double) MIN_PIXELS / ((long) height * width));
            hBar = (int) (Math.ceil(height * beta / FACTOR) * FACTOR);
            wBar = (int) (Math.ceil(width * beta / FACTOR) * FACTOR);
        }
        if ((long) hBar * wBar > MAX_PIXELS) {
            double beta = Math.sqrt((double) MAX_PIXELS / ((long) height * width));
            hBar = (int) (Math.floor(height * beta / FACTOR) * FACTOR);
            wBar = (int) (Math.floor(width * beta / FACTOR) * FACTOR);
        }
        return new int[]{hBar, wBar};
    }

    /**
     * Process an RGB Mat into pixel_values and grid_thw for the vision encoder.
     */
    public static ImageProcessResult process(Mat rgbMat, MatManager matManager) {
        int origH = rgbMat.rows();
        int origW = rgbMat.cols();
        int[] newSize = smartResize(origH, origW);
        int newH = newSize[0];
        int newW = newSize[1];

        Mat resized = matManager.newMat();
        Imgproc.resize(rgbMat, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_CUBIC);

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

        // Build [T, C, H, W] = [2, 3, H, W] - temporal duplication
        // Since both frames are identical, we can reference the same CHW data
        int gridT = 1;  // 2 frames / temporal_patch_size 2 = 1
        int gridH = H / PATCH_SIZE;
        int gridW = W / PATCH_SIZE;

        int ghMerge = gridH / MERGE_SIZE;
        int gwMerge = gridW / MERGE_SIZE;
        int numPatches = gridT * gridH * gridW;
        int patchDim = C * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE; // 1536

        // Extract patches matching transformers reshape+permute:
        // [T, C, H, W] -> [grid_t, temp, C, gh/merge, merge_h, patch_h, gw/merge, merge_w, patch_w]
        // -> permute(0,3,6,4,7,2,1,5,8) -> [grid_t, gh/merge, gw/merge, merge_h, merge_w, C, temp, patch_h, patch_w]
        // -> flatten to [num_patches, 1536]
        float[][] pixelValues = new float[numPatches][patchDim];

        int patchIdx = 0;
        for (int gt = 0; gt < gridT; gt++) {
            for (int ghm = 0; ghm < ghMerge; ghm++) {
                for (int gwm = 0; gwm < gwMerge; gwm++) {
                    for (int mh = 0; mh < MERGE_SIZE; mh++) {
                        for (int mw = 0; mw < MERGE_SIZE; mw++) {
                            int elemIdx = 0;
                            for (int c = 0; c < C; c++) {
                                for (int t = 0; t < TEMPORAL_PATCH_SIZE; t++) {
                                    // Both temporal frames are identical
                                    for (int ph = 0; ph < PATCH_SIZE; ph++) {
                                        int row = (ghm * MERGE_SIZE + mh) * PATCH_SIZE + ph;
                                        for (int pw = 0; pw < PATCH_SIZE; pw++) {
                                            int col = (gwm * MERGE_SIZE + mw) * PATCH_SIZE + pw;
                                            pixelValues[patchIdx][elemIdx++] = chw[c * H * W + row * W + col];
                                        }
                                    }
                                }
                            }
                            patchIdx++;
                        }
                    }
                }
            }
        }

        long[][] imageGridThw = new long[][]{{gridT, gridH, gridW}};
        int numMergedTokens = gridT * ghMerge * gwMerge * MERGE_SIZE * MERGE_SIZE;
        // numMergedTokens == gridT * gridH * gridW after merger: grid_t * (grid_h/merge) * (grid_w/merge)
        int numMergedAfterMerger = gridT * ghMerge * gwMerge;

        return new ImageProcessResult(pixelValues, imageGridThw, numMergedAfterMerger);
    }
}
