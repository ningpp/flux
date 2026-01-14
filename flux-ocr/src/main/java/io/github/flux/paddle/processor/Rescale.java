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
        Mat float64MatScaled = matManager.newMat();
        img.convertTo(float64MatScaled, CvType.CV_64FC3);
        img.release();
        Mat dest = matManager.newMat();
        Core.multiply(float64MatScaled, new Scalar(scale), dest);
        float64MatScaled.release();

        Mat float32MatScaled = matManager.newMat();
        dest.convertTo(float32MatScaled, CvType.CV_32FC3);
        dest.release();
        return float32MatScaled;
    }

}
