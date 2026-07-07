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

public class Rescale implements ImageProcessor {

    private final double scale;

    public Rescale(double scale) {
        this.scale = scale;
    }

    @Override
    public Mat process(MatManager matManager, Mat img) {
        // 直接转 float32 并乘以缩放因子，避免“先转 CV_64FC3 再转回 CV_32FC3”的两次转换，
        // 省去一个 ~9.8MB（640×640×3×8B）的临时分配。1/255 归一化在 float32 下的误差 < 1e-7，
        // 对模型输入精度无影响。
        Mat float32MatScaled = matManager.newMat();
        img.convertTo(float32MatScaled, CvType.CV_32FC3);
        Core.multiply(float32MatScaled, new Scalar(scale), float32MatScaled);
        matManager.release(img);
        return float32MatScaled;
    }

}
