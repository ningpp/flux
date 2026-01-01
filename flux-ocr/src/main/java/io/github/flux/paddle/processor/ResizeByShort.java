package io.github.flux.paddle.processor;

import ai.djl.modality.cv.Image.Interpolation;
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
    public Mat process(Mat img) {
        int h = img.rows();
        int w = img.cols();

        double scale = targetShortEdge / (double) Math.min(h, w);

        int rescaledHeight = (int) Math.round(h * scale);
        int rescaledWidth = (int) Math.round(w * scale);
        if (h == rescaledHeight && w == rescaledWidth) {
            return img;
        }
        Mat resized = new Mat();
        Imgproc.resize(img,
                resized, new Size(rescaledWidth, rescaledHeight),
                1, 1, interp.ordinal());
        return resized;
    }
}
