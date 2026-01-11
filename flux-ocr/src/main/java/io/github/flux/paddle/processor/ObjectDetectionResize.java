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

import io.github.flux.core.ObjectDetectionResizeResult;
import org.opencv.core.Mat;

public class ObjectDetectionResize {

    private final ResizeForObjectDect resize;

    public ObjectDetectionResize(ResizeForObjectDect resize) {
        this.resize = resize;
    }

    public ObjectDetectionResizeResult process(Mat img) {
        int[] ori_img_size = new int[] { img.cols(), img.rows() };
        Mat result_img = resize.process(img);
        int[] img_size = new int[] { result_img.cols(), result_img.rows() };
        double[] scale_factors = new double[] {
            img_size[0] / (double) ori_img_size[0],
            img_size[1] / (double) ori_img_size[1]
        };
        img.release();
        return new ObjectDetectionResizeResult(ori_img_size, result_img, img_size, scale_factors);
    }

}
