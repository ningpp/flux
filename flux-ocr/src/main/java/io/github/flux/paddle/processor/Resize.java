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

import ai.djl.modality.cv.Image.Interpolation;
import io.github.flux.core.MatManager;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Objects;

public class Resize implements ImageProcessor {
    private final int width;
    private final int height;
    protected final int interp;

    public Resize(int width, int height, Interpolation interp) {
        Objects.requireNonNull(interp, "you should set Interpolation");
        this.width = width;
        this.height = height;
        this.interp = interp.ordinal();
    }

    public Resize(int width, int height, int interp) {
        this.width = width;
        this.height = height;
        this.interp = interp;
    }

    @Override
    public Mat process(MatManager matManager, Mat img) {
        int h = img.rows();
        int w = img.cols();
        if (h == height && w == width) {
            return img;
        }
        Mat resized = matManager.newMat();
        Imgproc.resize(img,
                resized, new Size(width, height),
                1, 1, interp);
        matManager.release(img);
        return resized;
    }

}
