package io.github.flux.paddle.processor;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class Rescale implements ImageProcessor {

    private final double scale;

    public Rescale(double scale) {
        this.scale = scale;
    }

    @Override
    public Mat process(Mat img) {
        Mat float64MatScaled = new Mat();
        img.convertTo(float64MatScaled, CvType.CV_64FC3);
        img.release();
        Mat dest = new Mat();
        Core.multiply(float64MatScaled, new Scalar(scale), dest);
        float64MatScaled.release();

        Mat float32MatScaled = new Mat();
        dest.convertTo(float32MatScaled, CvType.CV_32FC3);
        dest.release();
        return float32MatScaled;
    }

}
