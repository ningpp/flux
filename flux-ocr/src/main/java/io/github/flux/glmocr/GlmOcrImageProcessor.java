/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.glmocr;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Image processor for GLM-OCR model.
 * 
 * GLM-OCR uses a unique image processing pipeline that:
 * 1. Resizes image to specific dimensions based on patch size (14x14)
 * 2. Splits image into patches of size 14x14
 * 3. Flattens each patch to [3*14*14=588] and normalizes
 * 4. Returns [num_patches, 1176] tensor (588 * 2 due to some internal processing)
 * 
 * The processor also computes image_grid_thw [t, h, w] for position embeddings.
 * 
 * This implementation provides a simplified Java version that mimics
 * the HuggingFace processor behavior for GLM-OCR.
 */
public class GlmOcrImageProcessor {

    // Patch configuration for GLM-OCR (from Python: image_processing_glm46v.py)
    private static final int PATCH_SIZE = 14;
    private static final int TEMPORAL_PATCH_SIZE = 2;
    private static final int MERGE_SIZE = 2;
    private static final int FACTOR = PATCH_SIZE * MERGE_SIZE;  // 28
    
    // From Python smart_resize: min_pixels = 112 * 112, max_pixels = 14 * 14 * 2 * 2 * 2 * 6144
    private static final int MIN_PIXELS = 112 * 112;  // 12544
    private static final int MAX_PIXELS = 14 * 14 * 2 * 2 * 2 * 6144;  // 9633792
    
    // Feature size per patch: 3 channels * 14 * 14 = 588
    // GLM-OCR uses 1176 = 588 * 2 (possibly due to temporal patch size)
    private static final int PATCH_FEATURES = 1176;
    
    // Normalization parameters (CLIP-style)
    private static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] STD = {0.26862954f, 0.26130258f, 0.27577711f};
    private static final float RESCALE_FACTOR = 1.0f / 255.0f;

    /**
     * Result of image preprocessing containing pixel values and grid info.
     */
    public static class PreprocessResult {
        public final float[][] pixelValues;  // [num_patches, 1176]
        public final int[] imageGridThw;     // [t, h, w]
        
        public PreprocessResult(float[][] pixelValues, int[] imageGridThw) {
            this.pixelValues = pixelValues;
            this.imageGridThw = imageGridThw;
        }
    }

    /**
     * Process a single RGB image for GLM-OCR vision encoder.
     *
     * @param rgbMat input RGB image
     * @param matManager OpenCV Mat resource manager
     * @param ndManager NDArray manager (unused in current implementation)
     * @return PreprocessResult with pixel_values and image_grid_thw
     */
    public static PreprocessResult process(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        // Calculate target size based on GLM-OCR constraints
        int origHeight = rgbMat.rows();
        int origWidth = rgbMat.cols();
        
        // Smart resize to fit within constraints
        int[] targetSize = smartResize(origHeight, origWidth);
        int targetHeight = targetSize[0];
        int targetWidth = targetSize[1];
        
        // Resize image using PIL-compatible BICUBIC (Catmull-Rom, a=-0.5) to match Python exactly
        Mat resized = pilBicubicResize(rgbMat, targetHeight, targetWidth, matManager);
        
        // Convert to float and normalize
        Mat floatMat = matManager.newMat();
        resized.convertTo(floatMat, CvType.CV_32FC3, RESCALE_FACTOR);
        
        // Calculate grid dimensions
        int hPatches = targetHeight / PATCH_SIZE;
        int wPatches = targetWidth / PATCH_SIZE;
        int temporal = 1;  // Single frame for static images
        
        // Create pixel_values tensor
        // For GLM-OCR, we need [num_patches, 1176] format
        int numPatches = temporal * hPatches * wPatches;
        float[][] pixelValues = new float[numPatches][PATCH_FEATURES];
        
        // Extract and normalize patches in spatial merge order
        // Python uses reshape + permute to reorder patches for spatial merging
        // The order is: for each 2x2 block, emit patches in [0,0], [0,1], [1,0], [1,1] order
        int hMerged = hPatches / MERGE_SIZE;
        int wMerged = wPatches / MERGE_SIZE;
        
        int patchIdx = 0;
        for (int t = 0; t < temporal; t++) {
            for (int i0 = 0; i0 < hMerged; i0++) {          // Block row
                for (int i2 = 0; i2 < wMerged; i2++) {      // Block col
                    for (int i1 = 0; i1 < MERGE_SIZE; i1++) {       // Within-block row
                        for (int i3 = 0; i3 < MERGE_SIZE; i3++) {   // Within-block col
                            int ph = i0 * MERGE_SIZE + i1;
                            int pw = i2 * MERGE_SIZE + i3;
                            extractPatch(floatMat, ph, pw, pixelValues[patchIdx]);
                            patchIdx++;
                        }
                    }
                }
            }
        }
        
        int[] imageGridThw = new int[]{temporal, hPatches, wPatches};
        
        return new PreprocessResult(pixelValues, imageGridThw);
    }
    
    /**
     * Smart resize matching Python's smart_resize from image_processing_glm46v.py
     * 
     * Python calls this with:
     *   num_frames=temporal_patch_size (2)
     *   temporal_factor=temporal_patch_size (2)
     *   factor=patch_size * merge_size (28)
     *   min_pixels=size["shortest_edge"] (12544)
     *   max_pixels=size["longest_edge"] (9633792)
     */
    private static int[] smartResize(int height, int width) {
        // Python uses num_frames=temporal_patch_size for the smart_resize call
        int numFrames = TEMPORAL_PATCH_SIZE;  // 2, not 1!
        
        // Ensure minimum size (from Python: if height < factor or width < factor)
        if (height < FACTOR || width < FACTOR) {
            double scale = Math.max((double) FACTOR / height, (double) FACTOR / width);
            height = (int) (height * scale);
            width = (int) (width * scale);
        }
        
        // Round to factor (using Math.round like Python)
        int hBar = (int) Math.round((double) height / FACTOR) * FACTOR;
        int wBar = (int) Math.round((double) width / FACTOR) * FACTOR;
        int tBar = (int) Math.round((double) numFrames / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE;
        
        // Scale if outside pixel constraints
        long totalPixels = (long) tBar * hBar * wBar;
        if (totalPixels > MAX_PIXELS) {
            double beta = Math.sqrt((double) (numFrames * height * width) / MAX_PIXELS);
            hBar = Math.max(FACTOR, (int) Math.floor(height / beta / FACTOR) * FACTOR);
            wBar = Math.max(FACTOR, (int) Math.floor(width / beta / FACTOR) * FACTOR);
        } else if (totalPixels < MIN_PIXELS) {
            double beta = Math.sqrt((double) MIN_PIXELS / (numFrames * height * width));
            hBar = (int) Math.ceil(height * beta / FACTOR) * FACTOR;
            wBar = (int) Math.ceil(width * beta / FACTOR) * FACTOR;
        }
        
        return new int[]{hBar, wBar};
    }
    
    /**
     * Extract a single patch and normalize it.
     * 
     * GLM-OCR expects patches in format: [R_t1, R_t2, G_t1, G_t2, B_t1, B_t2]
     * where each section is 14*14=196 values, and t1/t2 are temporal frames.
     * For static images, t1 and t2 are identical (duplicated).
     * Total: 6 * 196 = 1176 values per patch.
     * 
     * @param floatMat source image in float format [H, W, C]
     * @param patchH patch row index
     * @param patchW patch column index
     * @param output output array of size [PATCH_FEATURES] = 1176
     */
    private static void extractPatch(Mat floatMat, int patchH, int patchW, float[] output) {
        int startH = patchH * PATCH_SIZE;
        int startW = patchW * PATCH_SIZE;
        
        // Precompute normalized values for each pixel in the patch
        // Layout: [R_t1(196), R_t2(196), G_t1(196), G_t2(196), B_t1(196), B_t2(196)]
        float[] pixels = new float[3];
        int patchPixels = PATCH_SIZE * PATCH_SIZE;  // 196
        
        int idx = 0;
        for (int c = 0; c < 3; c++) {  // R, G, B channels
            // Temporal frame 1
            for (int h = 0; h < PATCH_SIZE; h++) {
                for (int w = 0; w < PATCH_SIZE; w++) {
                    floatMat.get(startH + h, startW + w, pixels);
                    float val = (pixels[c] - MEAN[c]) / STD[c];
                    output[idx++] = val;
                }
            }
            // Temporal frame 2 (duplicate for static image)
            for (int h = 0; h < PATCH_SIZE; h++) {
                for (int w = 0; w < PATCH_SIZE; w++) {
                    floatMat.get(startH + h, startW + w, pixels);
                    float val = (pixels[c] - MEAN[c]) / STD[c];
                    output[idx++] = val;
                }
            }
        }
    }

    // ====================================================================
    // PIL-compatible BICUBIC resize implementation
    // ====================================================================

    /**
     * Catmull-Rom cubic kernel (Keys, a=-0.5) matching PIL's BICUBIC.
     */
    private static double cubicKernel(double x) {
        x = Math.abs(x);
        if (x < 1.0) {
            return ((1.5 * x - 2.5) * x) * x + 1.0;
        }
        if (x < 2.0) {
            return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
        }
        return 0.0;
    }

    /**
     * Resize image using PIL-compatible BICUBIC (Catmull-Rom) interpolation.
     * This matches Python's PIL.Image.resize(size, Image.Resampling.BICUBIC).
     * 
     * Uses separable filtering: first horizontal, then vertical.
     * Includes anti-aliasing for downscaling (kernel support expanded by 1/scale).
     * Operates on uint8 values and rounds results to match PIL's behavior.
     *
     * @param src     source image (CV_8UC3, RGB)
     * @param newH    target height
     * @param newW    target width
     * @param matManager Mat manager for memory management
     * @return resized image (CV_8UC3, RGB)
     */
    private static Mat pilBicubicResize(Mat src, int newH, int newW, MatManager matManager) {
        int oldH = src.rows();
        int oldW = src.cols();

        // If no resize needed, return the original
        if (oldH == newH && oldW == newW) {
            return src;
        }

        // Step 1: horizontal resize (width)
        byte[] srcPixel = new byte[3];
        int[][] temp;  // intermediate storage [newH_or_oldH][newW][3]

        if (oldW != newW) {
            temp = new int[oldH][newW * 3];
            resizeDim(src, temp, oldH, oldW, newW, false);
        } else {
            // Copy src to temp (no horizontal resize needed)
            temp = new int[oldH][oldW * 3];
            for (int y = 0; y < oldH; y++) {
                for (int x = 0; x < oldW; x++) {
                    src.get(y, x, srcPixel);
                    temp[y][x * 3] = srcPixel[0] & 0xFF;
                    temp[y][x * 3 + 1] = srcPixel[1] & 0xFF;
                    temp[y][x * 3 + 2] = srcPixel[2] & 0xFF;
                }
            }
        }

        // Step 2: vertical resize (height)
        int interW = (oldW != newW) ? newW : oldW;
        int[][] result;
        if (oldH != newH) {
            result = new int[newH][interW * 3];
            resizeDimVertical(temp, result, oldH, newH, interW);
        } else {
            result = temp;
        }

        // Convert to Mat
        Mat dst = matManager.newMat();
        dst.create(newH, newW, CvType.CV_8UC3);
        byte[] px = new byte[3];
        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                px[0] = (byte) result[y][x * 3];
                px[1] = (byte) result[y][x * 3 + 1];
                px[2] = (byte) result[y][x * 3 + 2];
                dst.put(y, x, px);
            }
        }

        return dst;
    }

    /**
     * Resize horizontally using PIL-compatible BICUBIC.
     */
    private static void resizeDim(Mat src, int[][] dst, int height, int srcW, int dstW, boolean dummy) {
        double filterScale = Math.max(1.0, (double) srcW / dstW);
        double support = filterScale * 2.0;  // BICUBIC support = 2.0
        byte[] pixel = new byte[3];

        for (int y = 0; y < height; y++) {
            for (int outX = 0; outX < dstW; outX++) {
                double center = (outX + 0.5) * srcW / (double) dstW - 0.5;
                int start = (int) Math.floor(center - support);
                int stop = (int) Math.ceil(center + support);

                double[] sum = new double[3];
                double totalWeight = 0;

                for (int srcX = start; srcX <= stop; srcX++) {
                    double x = (srcX - center) / filterScale;
                    double w = cubicKernel(x);
                    if (w == 0.0) continue;

                    int clampedX = Math.max(0, Math.min(srcX, srcW - 1));
                    src.get(y, clampedX, pixel);
                    sum[0] += w * (pixel[0] & 0xFF);
                    sum[1] += w * (pixel[1] & 0xFF);
                    sum[2] += w * (pixel[2] & 0xFF);
                    totalWeight += w;
                }

                if (totalWeight > 0) {
                    dst[y][outX * 3] = clampByte(sum[0] / totalWeight);
                    dst[y][outX * 3 + 1] = clampByte(sum[1] / totalWeight);
                    dst[y][outX * 3 + 2] = clampByte(sum[2] / totalWeight);
                }
            }
        }
    }

    /**
     * Resize vertically using PIL-compatible BICUBIC.
     * Input and output are int arrays [height][width*3].
     */
    private static void resizeDimVertical(int[][] src, int[][] dst, int srcH, int dstH, int width) {
        double filterScale = Math.max(1.0, (double) srcH / dstH);
        double support = filterScale * 2.0;  // BICUBIC support = 2.0

        for (int outY = 0; outY < dstH; outY++) {
            double center = (outY + 0.5) * srcH / (double) dstH - 0.5;
            int start = (int) Math.floor(center - support);
            int stop = (int) Math.ceil(center + support);

            // Pre-compute weights for this row
            double[] weights = new double[stop - start + 1];
            double totalWeight = 0;
            for (int i = 0; i < weights.length; i++) {
                double y = ((start + i) - center) / filterScale;
                weights[i] = cubicKernel(y);
                totalWeight += weights[i];
            }
            // Normalize weights
            if (totalWeight > 0) {
                for (int i = 0; i < weights.length; i++) {
                    weights[i] /= totalWeight;
                }
            }

            for (int x = 0; x < width; x++) {
                double[] sum = new double[3];
                for (int i = 0; i < weights.length; i++) {
                    if (weights[i] == 0.0) continue;
                    int srcY = Math.max(0, Math.min(start + i, srcH - 1));
                    sum[0] += weights[i] * src[srcY][x * 3];
                    sum[1] += weights[i] * src[srcY][x * 3 + 1];
                    sum[2] += weights[i] * src[srcY][x * 3 + 2];
                }
                dst[outY][x * 3] = clampByte(sum[0]);
                dst[outY][x * 3 + 1] = clampByte(sum[1]);
                dst[outY][x * 3 + 2] = clampByte(sum[2]);
            }
        }
    }

    /**
     * Clamp and round a double value to [0, 255].
     */
    private static int clampByte(double value) {
        int v = (int) Math.round(value);
        return Math.max(0, Math.min(255, v));
    }
    
    /**
     * Convert preprocessed result to NDArray format for compatibility.
     */
    public static NDArray toNDArray(PreprocessResult result, NDManager ndManager) {
        int numPatches = result.pixelValues.length;
        int features = result.pixelValues[0].length;
        
        float[] flat = new float[numPatches * features];
        int idx = 0;
        for (int i = 0; i < numPatches; i++) {
            for (int j = 0; j < features; j++) {
                flat[idx++] = result.pixelValues[i][j];
            }
        }
        
        return ndManager.create(flat, new Shape(numPatches, features));
    }
}
