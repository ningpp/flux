package io.github.flux.paddle.processor;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Reorder the dimensions of the image from HWC to CHW.
 */
public class ToCHWImage implements ImageProcessor {

    @Override
    public Mat process(Mat img) {

        // Split channels
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        int height = img.rows();
        int width = img.cols();
        int channelSize = height * width;

        float[] chwData = new float[3 * channelSize];

        for (int c = 0; c < 3; c++) {
            Mat channel = channels.get(c); // shape: 224×224
            float[] channelData = new float[channelSize];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, chwData, c * channelSize, channelSize);
            channel.release();
        }

        // Create output Mat: size [3, 224, 224]
        Mat chw = new Mat(height, width, CvType.CV_32FC3);
        chw.put(0, 0, chwData);

        return chw;
    }

}
