package io.github.flux.paddle.processor;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ResizeForObjectDect implements ImageProcessor {
    private final int width;
    private final int height;
    protected final int interp;

    public ResizeForObjectDect(int width, int height, int interp) {
        this.width = width;
        this.height = height;
        this.interp = interp;
    }

    @Override
    public Mat process(Mat img) {
        int h = img.rows();
        int w = img.cols();
        if (h == height && w == width) {
            return img;
        }
        Mat resized = new Mat();
        Imgproc.resize(img,
                resized, new Size(width, height),
                1, 1, interp);
        img.release();
        return resized;
    }
}
