// this code is convert from  https://github.com/breezedeus/Pix2Text
// Pix2Text IS Licensed under the MIT License
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
package io.github.flux.formula.pix2text;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.MatManager;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class Pix2TextPreProcessor {

    public NDArray process(MatManager matManager, Mat rgbImg, NDManager manager) {
        DeiTImageProcessor deitProcessor = DeiTImageProcessor.builder()
                .setDoResize(true)
                .setSize(384, 384)
                .setDoCenterCrop(false)
                .setCropSize(224, 224)
                .setDoRescale(true)
                .setRescaleFactor(1.0f / 255.0f)
                .setDoNormalize(true)
                .setImageMean(new float[]{0.5f, 0.5f, 0.5f})
                .setImageStd(new float[]{0.5f, 0.5f, 0.5f})
                // 在 Pillow（PIL）中，Image.BICUBIC 表示三次卷积插值（Bicubic interpolation），其值为 Image.BICUBIC = 3
                // 在 OpenCV 中，对应的是：cv2.INTER_CUBIC  # 值为 2
                .setResampleMode(Imgproc.INTER_CUBIC)
                .build();

        NDArray imgNdArray = ImageUtil.toNDArrayUint8(rgbImg, manager);
        NDArray input = deitProcessor.preprocess(imgNdArray, matManager, manager);

        rgbImg.release();
        imgNdArray.close();
        return input;
    }

}
