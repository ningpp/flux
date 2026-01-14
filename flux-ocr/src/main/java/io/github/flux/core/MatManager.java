package io.github.flux.core;

import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MatManager implements AutoCloseable {

    private final Map<Long, Mat> resources = new ConcurrentHashMap<>();

    public Mat imread(String filename) {
        Mat mat = Imgcodecs.imread(filename);
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    public Mat newMat() {
        Mat mat = new Mat();
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    public Mat newMat(Mat src, Rect rect) {
        Mat mat = new Mat(src, rect);
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    public Mat newMat(int rows, int cols, int type) {
        Mat mat = new Mat(rows, cols, type);
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    @Override
    public void close() throws Exception {
        resources.forEach((_, mat) -> IOUtil.close(mat));
    }

}
