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
import io.github.flux.core.MatManager;
import io.github.flux.paddle.processor.ResizeNdArray;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;

/**
 * Image processor for GLM-OCR model.
 * Preprocesses images for the vision encoder.
 * 
 * GLM-OCR uses:
 * - Resize to 1120x1120 (or dynamic based on config)
 * - Normalize with specific mean/std values
 * - Convert to CHW format
 */
public class GlmOcrImageProcessor {

    // Default image size for GLM-OCR
    private static final int DEFAULT_IMAGE_SIZE = 1120;
    
    // Normalization parameters (typical for CLIP-based vision encoders)
    private static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] STD = {0.26862954f, 0.26130258f, 0.27577711f};
    private static final float RESCALE_FACTOR = 1.0f / 255.0f;

    /**
     * Process a single RGB image for GLM-OCR vision encoder.
     *
     * @param rgbMat input RGB image
     * @param matManager OpenCV Mat resource manager
     * @param ndManager NDArray manager
     * @return preprocessed image tensor [C, H, W]
     */
    public static NDArray process(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        return process(rgbMat, matManager, ndManager, DEFAULT_IMAGE_SIZE);
    }

    /**
     * Process a single RGB image with custom size.
     *
     * @param rgbMat input RGB image
     * @param matManager OpenCV Mat resource manager
     * @param ndManager NDArray manager
     * @param imageSize target image size (both width and height)
     * @return preprocessed image tensor [C, H, W]
     */
    public static NDArray process(Mat rgbMat, MatManager matManager, NDManager ndManager, int imageSize) {
        // Convert Mat to NDArray [H, W, C]
        NDArray image = ImageUtil.rgbToNDArray(rgbMat, ndManager);
        
        // Resize to target size
        NDArray resized = new ResizeNdArray(imageSize, imageSize, 2).process(matManager, List.of(image)).getFirst();
        
        // Rescale pixel values from [0, 255] to [0, 1]
        NDArray rescaled = rescale(resized, RESCALE_FACTOR);
        
        // Normalize with mean and std
        NDArray normalized = normalize(rescaled, MEAN, STD, ndManager);
        
        // Transpose from [H, W, C] to [C, H, W]
        return normalized.transpose(2, 0, 1);
    }

    /**
     * Rescale pixel values.
     */
    private static NDArray rescale(NDArray image, float factor) {
        return image.mul(factor);
    }

    /**
     * Normalize image with mean and std.
     * normalized = (image - mean) / std
     */
    private static NDArray normalize(NDArray image, float[] mean, float[] std, NDManager ndManager) {
        NDArray meanArray = ndManager.create(mean);
        NDArray stdArray = ndManager.create(std);
        return image.sub(meanArray).div(stdArray);
    }
}
