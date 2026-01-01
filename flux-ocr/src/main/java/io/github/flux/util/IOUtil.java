package io.github.flux.util;

import ai.onnxruntime.OnnxTensorLike;
import io.github.flux.core.PreProcessResult;
import org.opencv.core.Mat;

import java.util.List;

public final class IOUtil {

    private IOUtil() {
    }

    public static void close(OnnxTensorLike tensor) {
        if (tensor != null && !tensor.isClosed()) {
            try {
                tensor.close();
            } catch (Exception e) {
                // ignore
            }
        }
    }

    public static void close(List<Mat> mats) {
        if (mats == null) {
            return;
        }
        for (Mat mat : mats) {
            close(mat);
        }
    }

    public static void close(Mat mat) {
        if (mat != null) {
            try {
                mat.release();
            } catch (Exception e) {
                // ignore
            }
        }
    }

    public static void close(AutoCloseable closeable) {
        try {
            if (closeable != null) {
                closeable.close();
            }
        } catch (Exception e) {
            // ignore
        }
    }

    public static void close(PreProcessResult ppr) {
        if (ppr == null) {
            return;
        }
        close(ppr.mat());
        close(ppr.ndArray());
    }
}
