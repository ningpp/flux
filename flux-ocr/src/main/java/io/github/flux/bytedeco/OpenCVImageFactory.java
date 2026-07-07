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
package io.github.flux.bytedeco;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Utils;
import io.github.flux.core.MatManager;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;

/** {@code OpenCVImageFactory} is a high performance implementation of {@link ImageFactory}. */
public class OpenCVImageFactory extends ImageFactory {

    static {
        Loader.load(opencv_java.class);
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disables coffee cup image showing up on macOS
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    private MatManager matManager;

    public OpenCVImageFactory(MatManager matManager) {
        this.matManager = matManager;
    }

    /** {@inheritDoc} */
    @Override
    public Image fromFile(Path path) throws IOException {
        // Load image without alpha channel
        Mat img = Imgcodecs.imread(path.toAbsolutePath().toString());
        if (img.empty()) {
            throw new IOException("Read image failed: " + path);
        }
        return new OpenCVImage(matManager, img);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromInputStream(InputStream is) throws IOException {
        byte[] buf = Utils.toByteArray(is);
        Mat mat = new MatOfByte(buf);
        Mat img = Imgcodecs.imdecode(mat, Imgcodecs.IMREAD_COLOR);
        if (img.empty()) {
            throw new IOException("Read image failed.");
        }
        return new OpenCVImage(matManager, img);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromImage(Object image) {
        return new OpenCVImage(matManager, (Mat) image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromNDArray(NDArray array) {
        Shape shape = array.getShape();
        if (shape.dimension() == 4) {
            throw new UnsupportedOperationException("Batch is not supported");
        }
        // toType 可能返回新数组（视图/临时），使用完数据后必须关闭，否则泄露。
        // 仅关闭“新创建”的临时数组，绝不关闭调用方传入的 array 本身。
        NDArray typed = array.toType(DataType.UINT8, false);
        boolean typedIsNew = typed != array;
        boolean grayScale = shape.get(0) == 1 || shape.get(2) == 1;
        if (grayScale) {
            // expected CHW
            int width = Math.toIntExact(shape.get(2));
            int height = Math.toIntExact(shape.get(1));
            Mat img = matManager.newMat(height, width, CvType.CV_8UC1);
            img.put(0, 0, typed.toByteArray());
            if (typedIsNew) {
                typed.close();
            }
            return new OpenCVImage(matManager, img);
        }
        // CHW 需要转成 HWC；transpose 返回视图，需在使用后关闭。
        NDArray transposed = typed;
        if (NDImageUtils.isCHW(typed.getShape())) {
            transposed = typed.transpose(1, 2, 0);
        }
        int width = Math.toIntExact(transposed.getShape().get(1));
        int height = Math.toIntExact(transposed.getShape().get(0));
        Mat img = matManager.newMat(height, width, CvType.CV_8UC3);
        img.put(0, 0, transposed.toByteArray());
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGR);
        if (transposed != typed) {
            transposed.close();
        }
        if (typedIsNew) {
            typed.close();
        }
        return new OpenCVImage(matManager, img);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromPixels(int[] pixels, int width, int height) {
        Mat img = matManager.newMat(height, width, CvType.CV_8UC4);
        byte[] data = new byte[width * height * 4];
        IntBuffer buf = ByteBuffer.wrap(data).asIntBuffer();
        for (int pixel : pixels) {
            int r = pixel >> 8 & 0xff00;
            int g = pixel << 8 & 0xff0000;
            int b = pixel << 24 & 0xff000000;
            buf.put(pixel >>> 24 | b | g | r);
        }
        img.put(0, 0, data);
        return new OpenCVImage(matManager, img);
    }
}
