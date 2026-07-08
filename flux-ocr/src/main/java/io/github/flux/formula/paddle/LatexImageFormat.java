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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import io.github.flux.bytedeco.OpenCVImageFactory;
import io.github.flux.core.MatManager;
import io.github.flux.util.IOUtil;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;

import java.util.ArrayList;
import java.util.List;

/**
 * Class for formatting images to a specific format suitable for LaTeX.
 */
public class LatexImageFormat implements ImageProcessor {

    public Mat ___process(MatManager matManager, Mat img) {
        // Get image dimensions
        int imH = img.rows();
        int imW = img.cols();

        // Calculate the new dimensions to be multiples of 16
        int divideH = (int) (Math.ceil((double) imH / 16.0) * 16);
        int divideW = (int) (Math.ceil((double) imW / 16.0) * 16);

        // Pad the image to the new dimensions if necessary.
        // This is equivalent to np.pad with constant_values=1.
        // Note: In the case of (384, 384) input, padding will be zero.
        Mat paddedImg = matManager.newMat();
        int top = 0;
        int bottom = divideH - imH;
        int left = 0;
        int right = divideW - imW;
        Core.copyMakeBorder(img, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(1, 1, 1));

        // Convert the image to a blob with NCHW layout.
        // The Dnn.blobFromImage function is perfect for this. It handles the
        // conversion from OpenCV's default HWC format to the NCHW format
        // required by many deep learning frameworks.
        // We set scalefactor to 1.0 because no normalization is in the python code.
        // We set swapRB to false to keep the channel order (e.g., BGR).
        Mat blob = Dnn.blobFromImage(paddedImg, 1.0, new Size(divideW, divideH), new Scalar(0,0,0), false, false);

        // The blob is 4-dimensional (N, C, H, W). In this case (1, 3, 384, 384).
        // We need to return a 3-dimensional Mat (C, H, W).
        // We can reshape the blob to achieve this.
        // The total number of elements is 1 * 3 * 384 * 384.
        // The new shape will be [3, 384, 384].
        int[] newShape = {3, paddedImg.rows(), paddedImg.cols()};
        Mat result = blob.reshape(1, newShape);

        // Release intermediate Mats to free memory
        matManager.release(paddedImg);
        IOUtil.close(blob);

        return result;
    }

    /**
     * Formats a single image to a format compatible with certain processing pipelines,
     * often used for deep learning models.
     *
     * @param img The input image as an OpenCV Mat object.
     * @return The formatted image as an OpenCV Mat object with dimensions reshaped to (1, 1, H, W).
     */
    // @Override
    public Mat process_(MatManager matManager, Mat img) {
        // Get image dimensions
        int imH = img.rows();
        int imW = img.cols();

        // Calculate the new height and width to be the smallest multiple of 16
        // that is greater than or equal to the original dimensions.
        int divideH = (int) (Math.ceil((double) imH / 16.0) * 16);
        int divideW = (int) (Math.ceil((double) imW / 16.0) * 16);

        // Corresponds to Python: img = img[:, :, 0]
        // This extracts the first channel of the image. If the image is already grayscale,
        // it will be a no-op in terms of channel data.
        Mat singleChannelImg = matManager.newMat();
        if (img.channels() > 1) {
            Core.extractChannel(img, singleChannelImg, 0);
        } else {
            singleChannelImg = img.clone();
        }

        // Corresponds to Python: np.pad(...)
        // Pad the image on the bottom and right to reach the new dimensions.
        // The padding value is 1, which assumes the image is in a floating-point format (e.g., 0.0-1.0).
        // If using an integer format (e.g., 0-255), a value of 255 might be more appropriate for white padding.
        Mat paddedImg = matManager.newMat();
        Core.copyMakeBorder(singleChannelImg, paddedImg, 0, divideH - imH, 0, divideW - imW, Core.BORDER_CONSTANT, new Scalar(1));

        // Corresponds to Python: img[:, :, np.newaxis].transpose(2, 0, 1)[np.newaxis, :]
        // This sequence of operations converts the 2D image (H, W) into a 4D tensor (1, 1, H, W),
        // which is a common input format (NCHW) for deep learning models.

        // 1. Create a List of Mats to represent the channels (C). Here, it's just one channel.
        List<Mat> channels = new ArrayList<>();
        channels.add(paddedImg);

        // 2. Merge the channels into a single Mat. This creates a (1, H, W) structure internally for dnn.blobFromImage
        Mat blob = Dnn.blobFromImage(paddedImg);

        return blob;
    }

    /**
     * Formats a single image to a format compatible with certain processing pipelines,
     * often used for deep learning models.
     *
     * @param matManager
     * @param img        The input image as an OpenCV Mat object.
     * @return The formatted image as an OpenCV Mat object with dimensions reshaped to (1, 1, H, W).
     */
    @Override
    public Mat process(MatManager matManager, Mat img) {
        var manager = NDManager.newBaseManager();
        NDArray imgNdArray = ImageUtil.toNDArray(matManager, img, manager, null);
        NDArray result = format(imgNdArray, manager);

        return (Mat) new OpenCVImageFactory(matManager).fromNDArray(result).getWrappedImage();
    }

    /**
     * Formats a single image to the LaTeX-compatible format.
     *
     * @param img The input image as an NDArray.
     * @return The formatted image as an NDArray with an added dimension for color.
     */
    public NDArray format(NDArray img, NDManager manager) {
        // Get image dimensions
        long imH = img.getShape().get(0);
        long imW = img.getShape().get(1);

        // Calculate new dimensions, making them divisible by 16
        long divideH = (long) Math.ceil(imH / 16.0) * 16;
        long divideW = (long) Math.ceil(imW / 16.0) * 16;

        // Take only the first channel of the image
        NDArray processedImg = img.get(":, :, 0");

        // Create a new array filled with 1s for padding
        NDArray paddedImg = manager.ones(new Shape(divideH, divideW));

        // Copy the original image data into the padded array
        paddedImg.set(new NDIndex("0:" + imH + ", 0:" + imW), processedImg);
        processedImg = paddedImg;

        // Expand dimensions and transpose to (channel, height, width)
        NDArray imgExpanded = processedImg.expandDims(2).transpose(2, 0, 1);

        return imgExpanded;
    }
}
