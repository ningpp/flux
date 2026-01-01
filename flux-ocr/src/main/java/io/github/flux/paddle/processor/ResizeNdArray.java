package io.github.flux.paddle.processor;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.Image.Interpolation;
import ai.djl.ndarray.NDArray;
import io.github.flux.bytedeco.OpenCVImageFactory;
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

    public List<NDArray> process(List<NDArray> imgs) {
        var res = new ArrayList<NDArray>();
        for (var img : imgs) {
            res.add(resize(img));
            img.close();
        }
        return res;
    }

    private NDArray resize(NDArray img) {
        long[] shape = img.getShape().getShape();
        int h = (int) shape[0];
        int w = (int) shape[1];
        if (h == height && w == width) {
            return img;
        }
        Image cv2Img = OpenCVImageFactory.INSTANCE.fromNDArray(img);
        Mat resized = new Mat();
        Imgproc.resize((Mat) cv2Img.getWrappedImage(),
                resized, new Size(width, height),
                1, 1, interp);
        // 这个和Python的结果不一样ai.djl.modality.cv.util.NDImageUtils.resize(img, rescaledWidth, rescaledHeight, interp)
        return ImageUtil.toNDArray(resized, img.getManager(), null);
    }

}
