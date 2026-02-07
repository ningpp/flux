package io.github.flux.qwen3vl;

import io.github.flux.core.MatManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Image preprocessing for Qwen3-VL.
 * Pipeline:
 *   1. constrained_resize — pick (H, W) from valid options where grid_h * grid_w == EXPORT_PATCHES,
 *      choosing the aspect ratio closest to the input image.
 *   2. Rescale ÷255, normalize with mean=0.5 std=0.5
 *   3. Temporal duplication (temporal_patch_size=2)
 *   4. Patch extraction with merge_size=2 grouping
 *
 * The ONNX vision encoder has position embeddings baked in for EXPORT_PATCHES total patches.
 * All images must be resized to produce exactly that many patches.
 *
 * Output: pixel_values [num_patches, 1536], image_grid_thw [1, 3].
 */
public class Qwen3VlImageProcessor {

    private static final int PATCH_SIZE = 16;
    private static final int TEMPORAL_PATCH_SIZE = 2;
    private static final int MERGE_SIZE = 2;

    /**
     * Total patch count the ONNX vision encoder was exported with.
     * Export grid [1, 12, 24] → 288 patches.  Change if you re-export.
     */
    private static final int EXPORT_PATCHES = 288;
    private static final int MIN_GRID_DIM = 4;  // minimum grid_h or grid_w (= 64 px)
    private static final int MAX_GRID_DIM = 24; // max(export_grid_h, export_grid_w) — rotary table size

    private static final float RESCALE_FACTOR = 1.0f / 255.0f;
    private static final float IMAGE_MEAN = 0.5f;
    private static final float IMAGE_STD = 0.5f;

    /** Pre-computed valid (pixelH, pixelW) options. */
    private static final int[][] VALID_SIZES;
    static {
        List<int[]> list = new ArrayList<>();
        for (int gh = MIN_GRID_DIM; gh <= Math.min(EXPORT_PATCHES, MAX_GRID_DIM); gh += MERGE_SIZE) {
            if (EXPORT_PATCHES % gh != 0) continue;
            int gw = EXPORT_PATCHES / gh;
            if (gw < MIN_GRID_DIM || gw > MAX_GRID_DIM || gw % MERGE_SIZE != 0) continue;
            list.add(new int[]{gh * PATCH_SIZE, gw * PATCH_SIZE});
        }
        VALID_SIZES = list.toArray(new int[0][]);
    }

    /**
     * Result of image preprocessing, ready for the vision encoder.
     */
    public record ImageProcessResult(
            float[][] pixelValues,    // [num_patches, 1536]
            long[][] imageGridThw,    // [1, 3] = {{grid_t, grid_h, grid_w}}
            int numMergedTokens       // grid_t * (grid_h / merge) * (grid_w / merge)
    ) {}

    /**
     * Pick the (H, W) from valid options that best matches the input aspect ratio.
     * The ONNX vision encoder only supports images producing exactly EXPORT_PATCHES.
     */
    public static int[] constrainedResize(int height, int width) {
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("Invalid image size: " + height + "x" + width);
        }
        double aspect = (double) width / height;
        int[] best = VALID_SIZES[0];
        double bestDiff = Double.MAX_VALUE;
        for (int[] hw : VALID_SIZES) {
            double diff = Math.abs((double) hw[1] / hw[0] - aspect);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = hw;
            }
        }
        return new int[]{best[0], best[1]};
    }

    /**
     * Process an RGB Mat into pixel_values and grid_thw for the vision encoder.
     */
    public static ImageProcessResult process(Mat rgbMat, MatManager matManager) {
        int origH = rgbMat.rows();
        int origW = rgbMat.cols();
        int[] newSize = constrainedResize(origH, origW);
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
