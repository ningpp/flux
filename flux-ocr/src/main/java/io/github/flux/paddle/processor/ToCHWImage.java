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
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Reorder the dimensions of the image from HWC to CHW.
 */
public class ToCHWImage implements ImageProcessor {

    @Override
    public Mat process(MatManager matManager, Mat img) {

        int height = img.rows();
        int width = img.cols();
        int channelSize = height * width;

        // 一次性读取 HWC 交错浮点数据，再在 Java 中重排为 CHW 平面布局。
        // 相比 split(3 个 native Mat) + 3 次 get + merge，减少 native 分配与 JNI 调用，
        // 同时保证写入 CHW Mat 的底层 buffer 与优化前逐通道拷贝的结果完全一致。
        int total = 3 * channelSize;
        float[] hwc = new float[total];
        img.get(0, 0, hwc);

        float[] chwData = new float[total];
        for (int h = 0; h < height; h++) {
            int rowBase = h * width;
            for (int w = 0; w < width; w++) {
                int src = (rowBase + w) * 3;
                int dst = rowBase + w;
                chwData[dst] = hwc[src];                       // R plane
                chwData[dst + channelSize] = hwc[src + 1];      // G plane
                chwData[dst + 2 * channelSize] = hwc[src + 2];  // B plane
            }
        }

        Mat chw = matManager.newMat(height, width, CvType.CV_32FC3);
        chw.put(0, 0, chwData);

        // 释放输入 Mat（上一步 Normalize 产生的临时 Mat），避免长期存活的 MatManager 累积
        matManager.release(img);

        return chw;
    }

}
