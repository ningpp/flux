package io.github.flux.paddle.processor;

import io.github.flux.core.ObjectDetectionResizeResult;
import org.opencv.core.Mat;

public class ObjectDetectionResize {

    private final ResizeForObjectDect resize;

    public ObjectDetectionResize(ResizeForObjectDect resize) {
        this.resize = resize;
    }

    public ObjectDetectionResizeResult process(Mat img) {
        int[] ori_img_size = new int[] { img.cols(), img.rows() };
        Mat result_img = resize.process(img);
        int[] img_size = new int[] { result_img.cols(), result_img.rows() };
        double[] scale_factors = new double[] {
            img_size[0] / (double) ori_img_size[0],
            img_size[1] / (double) ori_img_size[1]
        };
        img.release();
        return new ObjectDetectionResizeResult(ori_img_size, result_img, img_size, scale_factors);
    }

}
