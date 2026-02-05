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

    // Patch configuration for GLM-OCR
    private static final int PATCH_SIZE = 14;
    private static final int TEMPORAL_PATCH_SIZE = 2;
    private static final int MERGE_SIZE = 2;
    private static final int MIN_PIXELS = 4 * 28 * 28;  // Minimum image area
    private static final int MAX_PIXELS = 16384 * 28 * 28;  // Maximum image area
    
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
        
        // Extract and normalize patches
        int patchIdx = 0;
        for (int t = 0; t < temporal; t++) {
            for (int ph = 0; ph < hPatches; ph++) {
                for (int pw = 0; pw < wPatches; pw++) {
                    extractPatch(floatMat, ph, pw, pixelValues[patchIdx]);
                    patchIdx++;
                }
            }
        }
        
        int[] imageGridThw = new int[]{temporal, hPatches, wPatches};
        
        return new PreprocessResult(pixelValues, imageGridThw);
    }
    
    /**
     * Smart resize to fit within pixel constraints while maintaining aspect ratio.
     * Dimensions are rounded to be divisible by patch size.
     */
    private static int[] smartResize(int height, int width) {
        // Calculate scale to fit within max pixels
        int totalPixels = height * width;
        double scale = 1.0;
        
        if (totalPixels > MAX_PIXELS) {
            scale = Math.sqrt((double) MAX_PIXELS / totalPixels);
        } else if (totalPixels < MIN_PIXELS) {
            scale = Math.sqrt((double) MIN_PIXELS / totalPixels);
        }
        
        int newHeight = (int) Math.round(height * scale);
        int newWidth = (int) Math.round(width * scale);
        
        // Round to multiple of patch size
        newHeight = Math.max(PATCH_SIZE, (newHeight / PATCH_SIZE) * PATCH_SIZE);
        newWidth = Math.max(PATCH_SIZE, (newWidth / PATCH_SIZE) * PATCH_SIZE);
        
        // Also ensure divisible by merge size
        int patchMergeSize = PATCH_SIZE * MERGE_SIZE;
        newHeight = Math.max(patchMergeSize, (newHeight / patchMergeSize) * patchMergeSize);
        newWidth = Math.max(patchMergeSize, (newWidth / patchMergeSize) * patchMergeSize);
        
        return new int[]{newHeight, newWidth};
    }
    
    /**
     * Extract a single patch and normalize it.
     * 
     * @param floatMat source image in float format [H, W, C]
     * @param patchH patch row index
     * @param patchW patch column index
     * @param output output array of size [PATCH_FEATURES]
     */
    private static void extractPatch(Mat floatMat, int patchH, int patchW, float[] output) {
        int startH = patchH * PATCH_SIZE;
        int startW = patchW * PATCH_SIZE;
        
        // Extract patch pixels and normalize
        // GLM-OCR expects [C, H, W] format, flattened
        // We also need to fill 1176 features (possibly with padding or temporal duplication)
        int baseIdx = 0;
        float[] pixels = new float[3];
        
        // First pass: extract 588 features (3 * 14 * 14 in CHW order)
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < PATCH_SIZE; h++) {
                for (int w = 0; w < PATCH_SIZE; w++) {
                    floatMat.get(startH + h, startW + w, pixels);
                    float val = pixels[c];
                    // Normalize: (val - mean) / std
                    val = (val - MEAN[c]) / STD[c];
                    if (baseIdx < output.length) {
                        output[baseIdx++] = val;
                    }
                }
            }
        }
        
        // Second pass: duplicate for temporal dimension to reach 1176
        // This mimics the temporal_patch_size=2 behavior
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < PATCH_SIZE; h++) {
                for (int w = 0; w < PATCH_SIZE; w++) {
                    floatMat.get(startH + h, startW + w, pixels);
                    float val = pixels[c];
                    val = (val - MEAN[c]) / STD[c];
                    if (baseIdx < output.length) {
                        output[baseIdx++] = val;
                    }
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
