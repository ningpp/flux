// this code is convert from https://github.com/PaddlePaddle/PaddleX
// PaddleX's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.paddle.processor;

import io.github.flux.core.MatManager;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Gemini 2.5 Pro
 */
public class OCRResizeNormImg implements ImageProcessor {

    private final int[] recImageShape;
    private final int[] inputShape;
    private final int maxImgW;

    public OCRResizeNormImg() {
        this(new int[]{3, 48, 320}, null);
    }

    public OCRResizeNormImg(int[] recImageShape, int[] inputShape) {
        this.recImageShape = recImageShape;
        this.inputShape = inputShape;
        this.maxImgW = 3200;
    }

    /**
     * Resizes and normalizes the image.
     * @param img The input image as a Mat.
     * @param maxWhRatio The maximum width to height ratio.
     * @return The processed image as a Mat.
     */
    public Mat resizeNormImg(MatManager matManager, Mat img, double maxWhRatio) {
        int imgC = recImageShape[0];
        int imgH = recImageShape[1];
        int imgW = recImageShape[2];

        if (img.channels() != imgC) {
            throw new IllegalArgumentException("Input image channels must be " + imgC);
        }

        imgW = (int) (imgH * maxWhRatio);

        Mat resizedImage = matManager.newMat();
        int resizedW;

        if (imgW > this.maxImgW) {
            Imgproc.resize(img, resizedImage, new Size(this.maxImgW, imgH));
            resizedW = this.maxImgW;
            imgW = this.maxImgW;
        } else {
            int h = img.rows();
            int w = img.cols();
            double ratio = w / (double) h;
            if (Math.ceil(imgH * ratio) > imgW) {
                resizedW = imgW;
            } else {
                resizedW = (int) Math.ceil(imgH * ratio);
            }
            Imgproc.resize(img, resizedImage, new Size(resizedW, imgH));
        }

        Mat finalImage = matManager.newMat(imgH, imgW, CvType.CV_32FC3);
        float[] finalImageData = normalizeToChw(resizedImage, imgC, imgH, imgW, resizedW);
        finalImage.put(0, 0, finalImageData);

        matManager.release(resizedImage);
        matManager.release(img);

        // Reshape to (C, H, W)
        return finalImage;
    }

    static float[] normalizeToChw(Mat resizedImage, int imgC, int imgH, int imgW, int resizedW) {
        float[] finalImageData = new float[imgC * imgH * imgW];
        byte[] resizedImageData = new byte[imgH * resizedW * imgC];
        resizedImage.get(0, 0, resizedImageData);

        float scale = 2.0f / 255.0f;
        int channelSize = imgH * imgW;
        for (int h = 0; h < imgH; h++) {
            int srcRowOffset = h * resizedW * imgC;
            int dstRowOffset = h * imgW;
            for (int w = 0; w < resizedW; w++) {
                int srcOffset = srcRowOffset + w * imgC;
                int dstOffset = dstRowOffset + w;
                for (int c = 0; c < imgC; c++) {
                    int unsignedValue = resizedImageData[srcOffset + c] & 0xFF;
                    finalImageData[c * channelSize + dstOffset] = unsignedValue * scale - 1.0f;
                }
            }
        }
        return finalImageData;
    }

    @Override
    public Mat process(MatManager matManager, Mat img) {
        if (this.inputShape == null) {
            return resize(matManager, img);
        } else {
            return staticResize(matManager, img);
        }
    }

    /**
     * Dynamic resizing based on image aspect ratio.
     * @param img The input image.
     * @return The resized and normalized image.
     */
    public Mat resize(MatManager matManager, Mat img) {
        int imgH = recImageShape[1];
        int imgW = recImageShape[2];

        double maxWhRatio = (double) imgW / imgH;
        int h = img.rows();
        int w = img.cols();
        double whRatio = w * 1.0 / h;
        maxWhRatio = Math.max(maxWhRatio, whRatio);

        return resizeNormImg(matManager, img, maxWhRatio);
    }

    /**
     * Static resizing to a fixed shape.
     * @param img The input image.
     * @return The resized and normalized image.
     */
    public Mat staticResize(MatManager matManager, Mat img) {
        int imgC = inputShape[0];
        int imgH = inputShape[1];
        int imgW = inputShape[2];

        Mat resizedImage = matManager.newMat();
        Imgproc.resize(img, resizedImage, new Size(imgW, imgH));

        // Normalization
        resizedImage.convertTo(resizedImage, CvType.CV_32F, 1.0 / 255.0);
        Core.subtract(resizedImage, new Scalar(0.5, 0.5, 0.5), resizedImage);
        Core.divide(resizedImage, new Scalar(0.5, 0.5, 0.5), resizedImage);

        // Transpose from HWC to CHW if needed
        Mat finalImage = matManager.newMat(imgC * imgH * imgW, 1, CvType.CV_32F);
        float[] finalImageData = new float[imgC * imgH * imgW];
        float[] resizedImageData = new float[imgH * imgW * imgC];
        resizedImage.get(0, 0, resizedImageData);

        int offset = 0;
        for (int c = 0; c < imgC; c++) {
            for (int h = 0; h < imgH; h++) {
                for (int w = 0; w < imgW; w++) {
                    finalImageData[offset++] = resizedImageData[(h * imgW + w) * imgC + c];
                }
            }
        }
        finalImage.put(0, 0, finalImageData);

        Mat result = finalImage.reshape(1, new int[]{imgC, imgH, imgW});

        matManager.release(resizedImage);
        matManager.release(finalImage);
        matManager.release(img);
        return result;
    }

}
