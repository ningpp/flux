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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.Image.Interpolation;
import ai.djl.ndarray.NDArray;
import io.github.flux.bytedeco.OpenCVImageFactory;
import io.github.flux.core.MatManager;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ResizeNdArray {
    private final int width;
    private final int height;
    protected final int interp;

    public ResizeNdArray(int width, int height, Interpolation interp) {
        this(width, height, interp.ordinal());
    }

    public ResizeNdArray(int width, int height, int interp) {
        this.width = width;
        this.height = height;
        this.interp = interp;
    }

    public List<NDArray> process(MatManager matManager, List<NDArray> imgs) {
        var res = new ArrayList<NDArray>();
        for (var img : imgs) {
            NDArray resized = resize(matManager, img);
            res.add(resized);
            if (resized != img) {
                img.close();
            }
        }
        return res;
    }

    private NDArray resize(MatManager matManager, NDArray img) {
        long[] shape = img.getShape().getShape();
        int h = (int) shape[0];
        int w = (int) shape[1];
        if (h == height && w == width) {
            return img;
        }
        Image cv2Img = new OpenCVImageFactory(matManager).fromNDArray(img);
        Mat resized = matManager.newMat();
        Imgproc.resize((Mat) cv2Img.getWrappedImage(),
                resized, new Size(width, height),
                1, 1, interp);
        // 这个和Python的结果不一样ai.djl.modality.cv.util.NDImageUtils.resize(img, rescaledWidth, rescaledHeight, interp)
        NDArray ndArray = ImageUtil.toNDArray(matManager, resized, img.getManager(), null);
        // 必须用 matManager.release 而非 IOUtil.close(Mat)：后者只释放原生内存，
        // 不会把 Mat 从 MatManager 的跟踪表中移除，会导致 trackedMatCount 无界增长（内存泄露）。
        matManager.release((Mat) cv2Img.getWrappedImage());
        matManager.release(resized);
        // img 由调用方 process() 统一关闭（仅在 resize 创建了新 NDArray 时）。
        return ndArray;
    }

}
