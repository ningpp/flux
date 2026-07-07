package io.github.flux.core;

import io.github.flux.util.IOUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MatManager implements AutoCloseable {

    private final Map<Long, Mat> resources = new ConcurrentHashMap<>();

    public Mat imread(String filename) {
        Mat mat = Imgcodecs.imread(filename);
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    public Mat imread(String filename, int flags) {
        Mat mat = Imgcodecs.imread(filename, flags);
        resources.put(mat.getNativeObjAddr(), mat);
        return mat;
    }

    public List<Mat> split(Mat mat) {
        List<Mat> splited = new ArrayList<>(3);
        Core.split(mat, splited);
        for (Mat s : splited) {
            resources.put(s.getNativeObjAddr(), s);
        }
        return splited;
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

    public Mat cloneMat(Mat mat) {
        Mat cloned = mat.clone();
        return track(cloned);
    }

    public Mat newMat(Size size, int type, Scalar s) {
        Mat mat = new Mat(size, type, s);
        return track(mat);
    }

    public Mat track(Mat mat) {
        if (mat != null) {
            resources.put(mat.getNativeObjAddr(), mat);
        }
        return mat;
    }

    /**
     * 当前被跟踪（尚未释放）的 Mat 数量。
     * 用于内存泄露验证：长期存活的 MatManager 在每轮推理后该值应回归到基线。
     */
    public int trackedMatCount() {
        return resources.size();
    }

    public void release(Mat mat) {
        // Idempotent: only release if the Mat is still tracked.
        // Some ImageProcessor implementations release their input Mat internally
        // while callers may also attempt to release it; this prevents double-free.
        if (mat != null && resources.remove(mat.getNativeObjAddr()) != null) {
            IOUtil.close(mat);
        }
    }

    /**
     * 释放一批被跟踪的 Mat 并从跟踪表中移除。
     * 用于释放 {@link #split(Mat)} 产生的临时 Mat，避免跟踪表无限累积（内存泄露）。
     */
    public void releaseAll(List<Mat> mats) {
        if (mats == null) {
            return;
        }
        for (Mat mat : mats) {
            release(mat);
        }
    }

    @Override
    public void close() throws Exception {
        resources.forEach((_, mat) -> IOUtil.close(mat));
        resources.clear();
    }

}
