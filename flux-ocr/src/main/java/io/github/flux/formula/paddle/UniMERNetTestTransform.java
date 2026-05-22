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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import io.github.flux.paddle.processor.ImageProcessor;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * A class for transforming images according to UniMERNet test specifications.
 */
public class UniMERNetTestTransform implements ImageProcessor {

    private final int numOutputChannels;

    /**
     * Initializes the UniMERNetTestTransform class with a default of 3 output channels.
     */
    public UniMERNetTestTransform() {
        this(3); // Default value from the Python __init__
    }

    /**
     * Initializes the UniMERNetTestTransform class with a specified number of output channels.
     *
     * @param numOutputChannels The number of channels for the output image.
     */
    public UniMERNetTestTransform(int numOutputChannels) {
        this.numOutputChannels = numOutputChannels;
    }

    /**
     * Transforms a single image for UniMERNet testing.
     *
     * @param matManager
     * @param img        The input image as an OpenCV Mat object.
     * @return The transformed image as an OpenCV Mat object.
     */
    @Override
    public Mat process(MatManager matManager, Mat img) {
        // --- 1. Normalization ---
        // Corresponds to Python: img = (img.astype("float32") * scale - mean) / std

        // Convert image from 8-bit integer (0-255) to 32-bit float (0.0-1.0)
        // This handles both `astype("float32")` and `* scale` where scale is 1/255.0
        Mat floatImg = matManager.newMat();
        img.convertTo(floatImg, CvType.CV_32F, 1.0 / 255.0);

        // Define mean and std. In OpenCV, we can use Scalar for per-channel operations.
        Scalar mean = new Scalar(0.7931, 0.7931, 0.7931);
        Scalar std = new Scalar(0.1738, 0.1738, 0.1738);

        // Subtract mean and divide by std deviation
        Core.subtract(floatImg, mean, floatImg);
        Core.divide(floatImg, std, floatImg);

        // --- 2. Convert to Grayscale ---
        // Corresponds to Python: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Mat grayscaleImage = matManager.newMat();
        Imgproc.cvtColor(floatImg, grayscaleImage, Imgproc.COLOR_BGR2GRAY);

        // The np.squeeze step is not needed in Java, as cvtColor to GRAY
        // already produces a 2D (single-channel) Mat.

        // --- 3. Merge channels ---
        // Corresponds to Python: cv2.merge([squeezed] * self.num_output_channels)
        // Create a list containing the grayscale image multiple times
        List<Mat> channels = new ArrayList<>();
        for (int i = 0; i < this.numOutputChannels; i++) {
            channels.add(grayscaleImage);
        }

        // Merge the list of channels back into a multi-channel Mat
        Mat mergedImage = matManager.newMat();
        Core.merge(channels, mergedImage);

        return mergedImage;
    }

    /**
     * 为 UniMERNet 测试转换单个图像。
     *
     * @param manager The NDManager to create new NDArrays.
     * @param img     The input image as an NDArray with shape (H, W, C).
     * @return The transformed image as an NDArray.
     */
    public NDArray transform(NDManager manager, NDArray img) {
        // 1. 定义均值、标准差和缩放比例
        float[] meanValues = {0.7931f, 0.7931f, 0.7931f};
        float[] stdValues = {0.1738f, 0.1738f, 0.1738f};
        float scale = 1.0f / 255.0f;

        // 创建与numpy中 (1, 1, 3) 形状等效的NDArray以进行广播
        NDArray mean = manager.create(meanValues, new Shape(1, 1, 3));
        NDArray std = manager.create(stdValues, new Shape(1, 1, 3));

        // 2. 归一化图像
        // (img * scale - mean) / std
        NDArray normalizedImg = img.toType(DataType.FLOAT32, false)
                .mul(scale)
                .sub(mean)
                .div(std);

        // 3. 将BGR图像转换为灰度图像
        // OpenCV's cvtColor(COLOR_BGR2GRAY) uses the formula: Y = 0.299*R + 0.587*G + 0.114*B
        // For a BGR NDArray with shape (H, W, C), channels are ordered B, G, R.
        // So we need weights for B, G, R channels respectively.
        NDArray weights = manager.create(new float[]{0.114f, 0.587f, 0.299f});
        // 第一次的代码
        // NDArray grayscaleImage = normalizedImg.dot(weights); // (H, W, C) * (C) -> (H, W)
        // 第二次的代码
        NDArray grayscaleImage = normalizedImg.matMul(weights); // (H, W, C) * (C) -> (H, W)

        // 4. 将单通道灰度图像复制到3个通道
        // 首先，增加一个维度以使其形状变为 (H, W, 1)
        NDArray unsqueezed = grayscaleImage.expandDims(-1);

        // 然后，沿着最后一个轴（通道轴）将图像堆叠3次
        NDArray mergedImg = unsqueezed.repeat(2, this.numOutputChannels);

        return mergedImg;
    }

}
