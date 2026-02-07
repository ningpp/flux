package io.github.flux.blip;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class BlipImageProcessor {

    private static final int TARGET_SIZE = 384;
    private static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] STD = {0.26862954f, 0.26130258f, 0.27577711f};

    /**
     * Process an RGB Mat into C,H,W NDArray (normalized).
     */
    public static NDArray process(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        Mat resized = matManager.newMat();
        Imgproc.resize(rgbMat, resized, new Size(TARGET_SIZE, TARGET_SIZE), 0, 0, Imgproc.INTER_CUBIC);

        Mat floatMat = matManager.newMat();
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

        int rows = floatMat.rows();
        int cols = floatMat.cols();
        int channels = floatMat.channels();
        int area = rows * cols;

        float[] pixels = new float[area * channels];
        floatMat.get(0, 0, pixels);

        float[] chwData = new float[area * channels];

        for (int i = 0; i < area; i++) {
            // R
            float r = pixels[i * 3];
            chwData[i] = (r - MEAN[0]) / STD[0];
            
            // G
            float g = pixels[i * 3 + 1];
            chwData[area + i] = (g - MEAN[1]) / STD[1];
            
            // B
            float b = pixels[i * 3 + 2];
            chwData[2 * area + i] = (b - MEAN[2]) / STD[2];
        }

        return ndManager.create(chwData, new Shape(channels, rows, cols));
    }
}
