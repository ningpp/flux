// this code is convert from  https://github.com/breezedeus/Pix2Text
// Pix2Text IS Licensed under the MIT License
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
package io.github.flux.formula.pix2text;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ResizeNdArray;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeiT Image Processor for Java using DJL's NDArray and OpenCV.
 * <p>
 * This class provides image preprocessing functionality for DeiT (Data-efficient Image Transformer) models,
 * including resizing, center cropping, rescaling, and normalization operations.
 */
public class DeiTImageProcessor {


    // ImageNet standard mean and std values
    private static final float[] IMAGENET_STANDARD_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STANDARD_STD = {0.229f, 0.224f, 0.225f};

    private final boolean doResize;
    private final Map<String, Integer> size;
    private final int resampleMode;
    private final boolean doCenterCrop;
    private final Map<String, Integer> cropSize;
    private final boolean doRescale;
    private final float rescaleFactor;
    private final boolean doNormalize;
    private final float[] imageMean;
    private final float[] imageStd;

    public static class Builder {
        private boolean doResize = true;
        private Map<String, Integer> size = Map.of("height", 256, "width", 256);
        private int resampleMode = Imgproc.INTER_CUBIC;
        private boolean doCenterCrop = true;
        private Map<String, Integer> cropSize = Map.of("height", 224, "width", 224);
        private boolean doRescale = true;
        private float rescaleFactor = 1.0f / 255.0f;
        private boolean doNormalize = true;
        private float[] imageMean = IMAGENET_STANDARD_MEAN.clone();
        private float[] imageStd = IMAGENET_STANDARD_STD.clone();

        public Builder setDoResize(boolean doResize) {
            this.doResize = doResize;
            return this;
        }

        public Builder setSize(int height, int width) {
            this.size = Map.of("height", height, "width", width);
            return this;
        }

        public Builder setResampleMode(int resampleMode) {
            this.resampleMode = resampleMode;
            return this;
        }

        public Builder setDoCenterCrop(boolean doCenterCrop) {
            this.doCenterCrop = doCenterCrop;
            return this;
        }

        public Builder setCropSize(int height, int width) {
            this.cropSize = Map.of("height", height, "width", width);
            return this;
        }

        public Builder setDoRescale(boolean doRescale) {
            this.doRescale = doRescale;
            return this;
        }

        public Builder setRescaleFactor(float rescaleFactor) {
            this.rescaleFactor = rescaleFactor;
            return this;
        }

        public Builder setDoNormalize(boolean doNormalize) {
            this.doNormalize = doNormalize;
            return this;
        }

        public Builder setImageMean(float[] imageMean) {
            this.imageMean = imageMean.clone();
            return this;
        }

        public Builder setImageStd(float[] imageStd) {
            this.imageStd = imageStd.clone();
            return this;
        }

        public DeiTImageProcessor build() {
            return new DeiTImageProcessor(this);
        }
    }

    private DeiTImageProcessor(Builder builder) {
        this.doResize = builder.doResize;
        this.size = new HashMap<>(builder.size);
        this.resampleMode = builder.resampleMode;
        this.doCenterCrop = builder.doCenterCrop;
        this.cropSize = new HashMap<>(builder.cropSize);
        this.doRescale = builder.doRescale;
        this.rescaleFactor = builder.rescaleFactor;
        this.doNormalize = builder.doNormalize;
        this.imageMean = builder.imageMean.clone();
        this.imageStd = builder.imageStd.clone();
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Resize an image using OpenCV.
     *
     * @param image        Input image as NDArray
     * @param targetHeight Target height
     * @param targetWidth  Target width
     * @return Resized image as NDArray
     */
    public NDArray resize(MatManager matManager, NDArray image, int targetHeight, int targetWidth) {
        return new ResizeNdArray(targetWidth, targetHeight, resampleMode).process(matManager, List.of(image)).get(0);
    }
        /*
        public NDArray resize(NDArray image, int targetHeight, int targetWidth, NDManager manager) {
            // Convert NDArray to OpenCV Mat
            Mat cvImage = ndArrayToMat(image);

            // Resize using OpenCV
            Mat resized = matManager.newMat();
            Size newSize = new Size(targetWidth, targetHeight);
            Imgproc.resize(cvImage, resized, newSize, 0, 0, resampleMode);

            // Convert back to NDArray
            NDArray result = ImageUtil.toNDArrayFloat(resized, manager);

            // Clean up
            cvImage.release();
            resized.release();

            return result;
        }
         */

    /**
     * Center crop an image.
     *
     * @param image      Input image as NDArray
     * @param cropHeight Target crop height
     * @param cropWidth  Target crop width
     * @param manager    NDManager for memory management
     * @return Center cropped image as NDArray
     */
    public NDArray centerCrop(NDArray image, int cropHeight, int cropWidth, NDManager manager) {
        throw new FluxException("Not Suport Center Crop image!");
    }

    /**
     * Rescale image pixel values.
     *
     * @param image Input image as NDArray
     * @param scale Scale factor
     * @return Rescaled image as NDArray
     */
    public static NDArray rescale(NDArray image, float scale) {
        return image.mul(scale);
    }

    /**
     * Normalize image using mean and standard deviation.
     *
     * @param image   Input image as NDArray
     * @param mean    Mean values for each channel
     * @param std     Standard deviation values for each channel
     * @param manager NDManager for memory management
     * @return Normalized image as NDArray
     */
    public static NDArray normalize(NDArray image, float[] mean, float[] std, NDManager manager) {
        // Convert mean and std to NDArrays
        NDArray meanArray = manager.create(mean).reshape(1, 1, 3);
        NDArray stdArray = manager.create(std).reshape(1, 1, 3);

        // Normalize: (image - mean) / std
        return image.sub(meanArray).div(stdArray);
    }

    /**
     * Preprocess a single image or batch of images.
     *
     * @param images  Input images as NDArray (can be single image or batch)
     * @param manager NDManager for memory management
     * @return Preprocessed images as NDArray
     */
    public NDArray preprocess(NDArray images, MatManager matManager, NDManager manager) {
        return preprocess(images, matManager, manager, null);
    }

    /**
     * Preprocess images with custom parameters.
     *
     * @param imgNdArray Input images as NDArray
     * @param manager    NDManager for memory management
     * @param config     Custom configuration (can be null to use defaults)
     * @return Preprocessed images as NDArray
     */
    public NDArray preprocess(NDArray imgNdArray, MatManager matManager, NDManager manager, PreprocessConfig config) {
        // Use default config if none provided
        if (config == null) {
            config = new PreprocessConfig();
        }

        boolean doResize = config.doResize != null ? config.doResize : this.doResize;
        boolean doCenterCrop = config.doCenterCrop != null ? config.doCenterCrop : this.doCenterCrop;
        boolean doRescale = config.doRescale != null ? config.doRescale : this.doRescale;
        boolean doNormalize = config.doNormalize != null ? config.doNormalize : this.doNormalize;

        Map<String, Integer> size = config.size != null ? config.size : this.size;
        Map<String, Integer> cropSize = config.cropSize != null ? config.cropSize : this.cropSize;
        float rescaleFactor = config.rescaleFactor != null ? config.rescaleFactor : this.rescaleFactor;
        float[] imageMean = config.imageMean != null ? config.imageMean : this.imageMean;
        float[] imageStd = config.imageStd != null ? config.imageStd : this.imageStd;

        // Determine if input is a batch or single image
        Shape shape = imgNdArray.getShape();
        boolean isBatch = shape.dimension() == 4; // (batch, height, width, channels)

        if (!isBatch) {
            // Add batch dimension if single image
            imgNdArray = imgNdArray.expandDims(0);
        }

        NDList processedImages = new NDList();
        long batchSize = imgNdArray.getShape().get(0);

        for (int i = 0; i < batchSize; i++) {
            NDArray image = imgNdArray.get(i);

            // Apply transformations
            if (doResize) {
                image = resize(matManager, image, size.get("height"), size.get("width"));
            }

            if (doCenterCrop) {
                image = centerCrop(image, cropSize.get("height"), cropSize.get("width"), manager);
            }

            if (doRescale) {
                image = rescale(image, rescaleFactor);
            }

            if (doNormalize) {
                image = normalize(image, imageMean, imageStd, manager);
            }

            // Convert to channels-first format (C, H, W)
            image = image.transpose(2, 0, 1);

            processedImages.add(image);
        }

        // Stack all processed images
        NDArray result = NDArrays.stack(processedImages);

        // Remove batch dimension if input was single image
        if (!isBatch) {
            result = result.squeeze(0);
        }

        return result;
    }

    /**
     * Convert NDArray to OpenCV Mat.
     */
    private Mat ndArrayToMat(MatManager matManager, NDArray ndArray) {
        Shape shape = ndArray.getShape();
        int height = (int) shape.get(0);
        int width = (int) shape.get(1);
        int channels = (int) shape.get(2);

        // Convert to float array
        float[] data = ndArray.toFloatArray();

        // Create Mat
        Mat mat = matManager.newMat(height, width, CvType.CV_32FC(channels));
        mat.put(0, 0, data);

        return mat;
    }

    /**
     * Configuration class for preprocessing parameters.
     */
    public static class PreprocessConfig {
        public Boolean doResize;
        public Map<String, Integer> size;
        public Boolean doCenterCrop;
        public Map<String, Integer> cropSize;
        public Boolean doRescale;
        public Float rescaleFactor;
        public Boolean doNormalize;
        public float[] imageMean;
        public float[] imageStd;

        public PreprocessConfig setDoResize(boolean doResize) {
            this.doResize = doResize;
            return this;
        }

        public PreprocessConfig setSize(int height, int width) {
            this.size = Map.of("height", height, "width", width);
            return this;
        }

        public PreprocessConfig setDoCenterCrop(boolean doCenterCrop) {
            this.doCenterCrop = doCenterCrop;
            return this;
        }

        public PreprocessConfig setCropSize(int height, int width) {
            this.cropSize = Map.of("height", height, "width", width);
            return this;
        }

        public PreprocessConfig setDoRescale(boolean doRescale) {
            this.doRescale = doRescale;
            return this;
        }

        public PreprocessConfig setRescaleFactor(float rescaleFactor) {
            this.rescaleFactor = rescaleFactor;
            return this;
        }

        public PreprocessConfig setDoNormalize(boolean doNormalize) {
            this.doNormalize = doNormalize;
            return this;
        }

        public PreprocessConfig setImageMean(float[] imageMean) {
            this.imageMean = imageMean.clone();
            return this;
        }

        public PreprocessConfig setImageStd(float[] imageStd) {
            this.imageStd = imageStd.clone();
            return this;
        }
    }

    // Getters for configuration values
    public boolean isDoResize() {
        return doResize;
    }

    public Map<String, Integer> getSize() {
        return new HashMap<>(size);
    }

    public int getResampleMode() {
        return resampleMode;
    }

    public boolean isDoCenterCrop() {
        return doCenterCrop;
    }

    public Map<String, Integer> getCropSize() {
        return new HashMap<>(cropSize);
    }

    public boolean isDoRescale() {
        return doRescale;
    }

    public float getRescaleFactor() {
        return rescaleFactor;
    }

    public boolean isDoNormalize() {
        return doNormalize;
    }

    public float[] getImageMean() {
        return imageMean.clone();
    }

    public float[] getImageStd() {
        return imageStd.clone();
    }

}
