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
package io.github.flux.formula.paddle;

import io.github.flux.core.MatManager;
import io.github.flux.paddle.processor.ImageProcessor;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.security.SecureRandom;
import java.util.Random;

/**
 * Class for decoding images for UniMERNet, including cropping margins, resizing, and padding.
 * This is a Java conversion of the original Python class.
 */
public class UniMERNetImgDecode implements ImageProcessor {

    private final Size inputSize;
    private final boolean randomPadding;
    private final Random random = new SecureRandom();

    /**
     * Initializes the UniMERNetImgDecode class.
     *
     * @param inputSize     The desired input size (height, width) for the images.
     * @param randomPadding Whether to use random padding for resizing.
     */
    public UniMERNetImgDecode(Size inputSize, boolean randomPadding) {
        this.inputSize = inputSize;
        this.randomPadding = randomPadding;
    }

    /**
     * Crops the margin of the image based on grayscale thresholding.
     *
     * @param img The input image as an OpenCV Mat.
     * @return The cropped image as a Mat.
     */
    public Mat cropMargin(MatManager matManager, Mat img) {
        // Corresponds to Python: img.convert("L")
        Mat gray = matManager.newMat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        // Corresponds to Python: data.astype(np.uint8)
        // This is implicit in OpenCV's Mat unless specific types are used.

        // Corresponds to Python: max_val = data.max(), min_val = data.min()
        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(gray);
        int minVal = (int) minMaxResult.minVal;
        int maxVal = (int) minMaxResult.maxVal;

        if (maxVal == minVal) {
            return img; // Return original image if it's a solid color
        }

        // Corresponds to Python: (data - min_val) / (max_val - min_val) * 255
        Mat r1 = matManager.newMat();
        Core.add(gray, new Scalar(-minVal), r1);
        Mat r2 = matManager.newMat();
        Core.multiply(r1, new Scalar(1.0 / (double)(maxVal-minVal)), r2);
        Mat dataf = matManager.newMat();
        Core.multiply(r2, new Scalar(255), dataf);
        Mat data = matManager.newMat();
        dataf.convertTo(data, CvType.CV_8U);

        // Corresponds to Python: 255 * (data < 200).astype(np.uint8)
        // Mat normalized = matManager.newMat();
        // Core.normalize(data, normalized, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        Mat thresholded = matManager.newMat();
        Imgproc.threshold(data, thresholded, 199, 255, Imgproc.THRESH_BINARY_INV);

        // C++: enum ThresholdTypes (cv.ThresholdTypes)
        /* Imgproc
                THRESH_BINARY = 0,
                THRESH_BINARY_INV = 1,
                THRESH_TRUNC = 2,
                THRESH_TOZERO = 3,
                THRESH_TOZERO_INV = 4,
                THRESH_MASK = 7,
                THRESH_OTSU = 8,
                THRESH_TRIANGLE = 16;
        */
        // Corresponds to Python: cv2.findNonZero(gray)
        Mat nonZeroCoords = matManager.newMat();
        Core.findNonZero(thresholded, nonZeroCoords);

        if (nonZeroCoords.empty()) {
            return img; // Return original if no content is found after thresholding
        }

        // Corresponds to Python: cv2.boundingRect(coords)
        Rect boundingRect = Imgproc.boundingRect(nonZeroCoords);

        // Corresponds to Python: img.crop((a, b, w + a, h + b))
        return matManager.newMat(img, boundingRect);
    }

    /**
     * Crops the margin of the image based on grayscale thresholding.
     *
     * @param img The input image as an OpenCV Mat.
     * @return The cropped image as a Mat.
     */
    public Mat __1__cropMargin(MatManager matManager, Mat img) {
        // Corresponds to Python: img.convert("L")
        Mat gray = matManager.newMat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        // Corresponds to Python: data.astype(np.uint8)
        // This is implicit in OpenCV's Mat unless specific types are used.

        // Corresponds to Python: max_val = data.max(), min_val = data.min()
        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(gray);
        int minVal = (int) minMaxResult.minVal;
        int maxVal = (int) minMaxResult.maxVal;

        if (maxVal == minVal) {
            return img; // Return original image if it's a solid color
        }

        // Corresponds to Python: (data - min_val) / (max_val - min_val) * 255
        Mat normalized = matManager.newMat();
        Core.normalize(gray, normalized, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // Corresponds to Python: 255 * (data < 200).astype(np.uint8)
        Mat thresholded = matManager.newMat();
        Imgproc.threshold(normalized, thresholded, 200, 255, Imgproc.THRESH_BINARY_INV);

        // Corresponds to Python: cv2.findNonZero(gray)
        Mat nonZeroCoords = matManager.newMat();
        Core.findNonZero(thresholded, nonZeroCoords);

        if (nonZeroCoords.empty()) {
            return img; // Return original if no content is found after thresholding
        }

        // Corresponds to Python: cv2.boundingRect(coords)
        Rect boundingRect = Imgproc.boundingRect(nonZeroCoords);

        // Corresponds to Python: img.crop((a, b, w + a, h + b))
        return matManager.newMat(img, boundingRect);
    }

    /**
     * Resizes the image to the specified size.
     *
     * @param img  The input image as a Mat.
     * @param size The desired size for the smallest edge.
     * @return The resized image as a Mat.
     */
    public Mat resize_not_same_(MatManager matManager, Mat img, int size) {
        int image_height = img.rows();
        int image_width = img.cols();

        // The Python logic resizes based on the smallest edge.
        int h = image_height;
        int w = image_width;
        int shortDim = Math.min(w, h);
        int longDim = Math.max(w, h);

        int newShort = size;
        int newLong = (int) (newShort * (double) longDim / shortDim);

        int newW, newH;
        if (w <= h) {
            newW = newShort;
            newH = newLong;
        } else {
            newW = newLong;
            newH = newShort;
        }

        Mat resizedImg = matManager.newMat();
        Imgproc.resize(img, resizedImg, new Size(newW, newH), 1, 1, Imgproc.INTER_LINEAR);
        return resizedImg;
    }

    /**
     * Decodes the image by cropping margins, resizing, and adding padding.
     *
     * @param matManager
     * @param img        The input image as a Mat.
     * @return The decoded image as a Mat, or null on failure.
     */
    @Override
    public Mat process(MatManager matManager, Mat img) {
        if (img == null || img.empty()) {
            return null;
        }

        // Corresponds to Python: self.crop_margin(Image.fromarray(img).convert("RGB"))
        Mat croppedImg = cropMargin(matManager, img);

        if (croppedImg.height() == 0 || croppedImg.width() == 0) {
            return null;
        }

        // Corresponds to Python: self.resize(img, min(self.input_size))
        int minInputSize = (int) Math.min(this.inputSize.height, this.inputSize.width);
        Mat resizedImg = resize_not_same_(matManager, croppedImg, minInputSize);

        // Corresponds to Python: img.thumbnail((self.input_size[1], self.input_size[0]))
        double r_width = this.inputSize.width / resizedImg.width();
        double r_height = this.inputSize.height / resizedImg.height();
        double r = Math.min(r_width, r_height);
        if (r < 1.0) { // only shrink if necessary
            Size thumbnailSize = new Size(Math.round(resizedImg.width() * r), Math.round(resizedImg.height() * r));
            Imgproc.resize(resizedImg, resizedImg, thumbnailSize, 0, 0, Imgproc.INTER_AREA);
        }

        // Corresponds to Python: padding calculation and ImageOps.expand
        int deltaWidth = (int) (this.inputSize.width - resizedImg.width());
        int deltaHeight = (int) (this.inputSize.height - resizedImg.height());

        int top, bottom, left, right;
        if (this.randomPadding) {
            left = (deltaWidth > 0) ? random.nextInt(deltaWidth + 1) : 0;
            top = (deltaHeight > 0) ? random.nextInt(deltaHeight + 1) : 0;
        } else {
            // 地板除
            // 注意，原始Python代码中使用了 【delta_width // 2】 【delta_height // 2】，Java和Python对负数的处理不一样
            left = Math.floorDiv(deltaWidth, 2);
            top = Math.floorDiv(deltaHeight, 2);
        }
        right = deltaWidth - left;
        bottom = deltaHeight - top;

        Mat paddedImg = matManager.newMat();
        Core.copyMakeBorder(resizedImg, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

        return paddedImg;
    }

}
