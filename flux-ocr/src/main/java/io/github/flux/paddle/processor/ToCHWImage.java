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

import io.github.flux.util.IOUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Reorder the dimensions of the image from HWC to CHW.
 */
public class ToCHWImage implements ImageProcessor {

    @Override
    public Mat process(Mat img) {

        // Split channels
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        int height = img.rows();
        int width = img.cols();
        int channelSize = height * width;

        float[] chwData = new float[3 * channelSize];

        for (int c = 0; c < 3; c++) {
            Mat channel = channels.get(c);
            float[] channelData = new float[channelSize];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, chwData, c * channelSize, channelSize);
            IOUtil.close(channel);
        }

        Mat chw = new Mat(height, width, CvType.CV_32FC3);
        chw.put(0, 0, chwData);

        IOUtil.close(channels);
        IOUtil.close(img);
        return chw;
    }

}
