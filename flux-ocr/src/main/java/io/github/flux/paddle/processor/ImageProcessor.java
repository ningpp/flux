package io.github.flux.paddle.processor;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public interface ImageProcessor {

    default List<Mat> process(List<Mat> imgs) {
        Objects.requireNonNull(imgs, "imgs can't be null");
        List<Mat> results = new ArrayList<>(imgs.size());
        for (Mat srcImg : imgs) {
            Mat result = process(srcImg);
            results.add(result);
            if (result != srcImg) {
                srcImg.release();
            }
        }
        return results;
    }

    Mat process(Mat img);

}
