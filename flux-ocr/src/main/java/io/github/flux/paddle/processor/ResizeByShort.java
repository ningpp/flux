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

/**
 * 按短边指定长度，等比缩放图像。如果需要，结果尺寸再对齐 sizeDivisor。
 */
public class ResizeByShort implements ImageProcessor {
    private final int targetShortEdge;
    protected final Interpolation interp;

    /**
     * @param targetShortEdge  短边目标长度
     * @param interp           插值方法
     */
    public ResizeByShort(int targetShortEdge, Interpolation interp) {
        this.targetShortEdge = targetShortEdge;
        this.interp = interp;
    }

    @Override
    public Mat process(MatManager matManager, Mat img) {
        int h = img.rows();
        int w = img.cols();

        double scale = targetShortEdge / (double) Math.min(h, w);

        int rescaledHeight = (int) Math.round(h * scale);
        int rescaledWidth = (int) Math.round(w * scale);
        if (h == rescaledHeight && w == rescaledWidth) {
            return img;
        }
        Mat resized = matManager.newMat();
        Imgproc.resize(img,
                resized, new Size(rescaledWidth, rescaledHeight),
                1, 1, interp.ordinal());
        matManager.release(img);
        return resized;
    }
}
