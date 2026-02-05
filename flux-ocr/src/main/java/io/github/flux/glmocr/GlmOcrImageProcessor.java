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
        
        // Resize image
        Mat resized = matManager.newMat();
        Imgproc.resize(rgbMat, resized, new Size(targetWidth, targetHeight), 0, 0, Imgproc.INTER_AREA);
        
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
