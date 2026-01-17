// this code is convert from https://github.com/OleehyO/TexTeller
// TexTeller's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.texteller;

import io.github.flux.core.MatManager;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class TexTellerProcessor implements ImageProcessor {
    private static final int FIXED_IMG_SIZE = 448;

    @Override
    public Mat process(MatManager matManager, Mat img) {
        Mat trimed = ImageUtil.trimWhiteBorder(matManager, img);

        Mat gray = ImageUtil.rgbToGray(matManager, trimed);
        Mat chw = ImageUtil.toOneChannelCHW(matManager, gray);
        Mat resized = ImageUtil.resize(
            matManager, chw, FIXED_IMG_SIZE-1, FIXED_IMG_SIZE, true
        );

        // toFloat
        Mat floatImg = matManager.newMat();
        resized.convertTo(floatImg, CvType.CV_32F, 1.0 / 255.0);

        // normalize
        Core.subtract(floatImg, new Scalar(0.9545467), floatImg);
        Core.divide(floatImg, new Scalar(0.15394445), floatImg);

        // pad
        return ImageUtil.padding(matManager, floatImg, FIXED_IMG_SIZE);
    }

}
