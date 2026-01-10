// this code is convert from https://github.com/Topdu/OpenOCR
// OpenOCR's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.unirec;

import ai.djl.modality.cv.Image.Interpolation;
import io.github.flux.core.Processor;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.ToCHWImage;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class UnirecProcessor implements Processor<Mat, Mat> {

    @Override
    public Mat process(Mat input) {
        int[] widthHeight = _calculate_target_size(input.cols(), input.rows());
        Mat resized = new Resize(widthHeight[0], widthHeight[1], Interpolation.BICUBIC).process(input);

        Mat floatImg = new Mat();
        resized.convertTo(floatImg, CvType.CV_32F, 1.0 / 255.0);

        // Define mean and std. In OpenCV, we can use Scalar for per-channel operations.
        Scalar mean = new Scalar(0.5, 0.5, 0.5);
        Scalar std = new Scalar(0.5, 0.5, 0.5);
        Core.subtract(floatImg, mean, floatImg);
        Core.divide(floatImg, std, floatImg);
        return new ToCHWImage().process(floatImg);
    }

    // Calculate target size with aspect ratio preservation.
    private int[] _calculate_target_size(int original_width, int original_height) {
        int max_width = 960;
        int max_height = 1408;
        double aspect_ratio = original_width / (double) original_height;

        int new_height;
        int new_width;
        if (original_width > max_width || original_height > max_height) {
            if (max_width >= aspect_ratio * max_height) {
                new_height = max_height;
                new_width = Double.valueOf(new_height * aspect_ratio).intValue();
            } else {
                new_width = max_width;
                new_height = Double.valueOf(new_width / aspect_ratio).intValue();
            }
        } else {
            new_width = original_width;
            new_height = original_height;
        }

        // Apply divided factor
        int div_w = 64;
        int div_h = 64;

        int final_width = Math.max(Math.floorDiv(new_width, div_w)* div_w, 64);
        int final_height = Math.max(Math.floorDiv(new_height, div_h)* div_h, 64);
        return new int[] { final_width, final_height };
    }

}
