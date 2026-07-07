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
import org.opencv.core.Mat;
import org.opencv.core.Rect;

/**
 * Crop transform.
 */
public class Crop implements ImageProcessor {
    private final int cropW;
    private final int cropH;
    private final String mode;

    public Crop(int size) {
        this(size, size);
    }

    public Crop(int cropW, int cropH) {
        this(cropW, cropH, "C");
    }

    public Crop(int cropW, int cropH, String mode) {
        checkImageSize(new int[]{cropW, cropH});
        this.cropW = cropW;
        this.cropH = cropH;
        if (!mode.equals("C") && !mode.equals("TL")) {
            throw new IllegalArgumentException("Unsupported mode");
        }
        this.mode = mode;
    }


    /** 对单张 HWC 格式的 NDArray 做裁剪 */
    @Override
    public Mat process(MatManager matManager, Mat img) {
        // 假设输入是 [height, width, channel]
        int h = img.rows();
        int w = img.cols();
        if (w < cropW || h < cropH) {
            throw new IllegalArgumentException(
                    String.format("Input image (%d,%d) smaller than target (%d,%d)", w, h, cropW, cropH));
        }

        int x1, y1;
        if (mode.equals("C")) {
            // 地板除
            // 注意，原始Python代码中使用了(w - cw) // 2，Java和Python对负数的处理不一样
            x1 = Math.max(0, Math.floorDiv(w - cropW, 2));
            y1 = Math.max(0, Math.floorDiv(h - cropH, 2));
        } else { // "TL"
            x1 = 0;
            y1 = 0;
        }
        int x2 = Math.min(w, x1 + cropW);
        int y2 = Math.min(h, y1 + cropH);

        return slice(matManager, img, x1, y1, x2, y2);
    }

    public static Mat slice(MatManager matManager, Mat img, int x1, int y1, int x2, int y2) {
        // 通过 MatManager 创建子矩阵：语义等价于 img.submat，但子矩阵会被登记到
        // 跟踪表，且底层 buffer 与父 Mat 共享（引用计数 +1）。
        // 随后经由 MatManager 释放父 Mat：既把父 Mat 从跟踪表移除，又因子矩阵仍持有
        // 引用而保证底层 buffer 不会被提前释放；下游 Normalize 释放该子矩阵时 buffer
        // 才真正回收，从而消除“跟踪表幽灵条目 + 原生 buffer 泄露”两类问题。
        Mat result = matManager.newMat(img, new Rect(x1, y1, x2 - x1, y2 - y1));
        matManager.release(img);
        return result;
    }

    /** 简单校验：保证 size 长度为 2、且均为正数 */
    private static void checkImageSize(int[] size) {
        if (size.length != 2 || size[0] <= 0 || size[1] <= 0) {
            throw new IllegalArgumentException(
                    "crop_size must be length-2 positive ints, but got: "
                            + java.util.Arrays.toString(size));
        }
    }

}
