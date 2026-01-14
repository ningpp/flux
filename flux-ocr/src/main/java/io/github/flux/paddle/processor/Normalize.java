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
import io.github.flux.util.IOUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Normalize image channels.
 */
public class Normalize implements ImageProcessor {

    private final double[] alpha;
    private final double[] beta;

    /**
     * 默认构造：scale=1/255, mean=0.5, std=0.5
     */
    public Normalize() {
        this(1.0 / 255.0, new double[]{0.5, 0.5, 0.5}, new double[]{0.5, 0.5, 0.5});
    }

    /**
     * 构造函数，手动指定 scale、mean、std
     *
     * @param scale 缩放因子
     * @param mean  每个通道的均值，长度必须为 3
     * @param std   每个通道的标准差，长度必须为 3
     */
    public Normalize(double scale, double[] mean, double[] std) {
        if (mean.length != 3) {
            throw new IllegalArgumentException(
                    "Expected `mean` to be length 3, but got " + mean.length);
        }
        if (std.length != 3) {
            throw new IllegalArgumentException(
                    "Expected `std` to be length 3, but got " + std.length);
        }
        alpha = new double[3];
        beta = new double[3];
        for (int i = 0; i < 3; i++) {
            alpha[i] = scale / std[i];
            beta[i] = -mean[i] / std[i];
        }
    }

    @Override
    public Mat process(MatManager matManager, Mat mat) {
        Mat matFloat32 = matManager.newMat();
        mat.convertTo(matFloat32, CvType.CV_32F);
        List<Mat> splitIm = new ArrayList<>();
        Core.split(matFloat32, splitIm);

        for (int c = 0; c < mat.channels(); c++) {
            Mat channel = splitIm.get(c);

            Core.multiply(channel, new Scalar(alpha[c]), channel);

            Core.add(channel, new Scalar(beta[c]), channel);
        }

        Mat res = matManager.newMat();
        Core.merge(splitIm, res);
        IOUtil.close(splitIm);
        IOUtil.close(matFloat32);
        IOUtil.close(mat);
        return res;
    }
}
