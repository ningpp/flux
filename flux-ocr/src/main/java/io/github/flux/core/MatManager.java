package io.github.flux.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;

public class MatManager implements AutoCloseable {

    private final Map<Long, Mat> resources = new ConcurrentHashMap<>();
    private final ConcurrentLinkedDeque<AutoCloseable> closeables = new ConcurrentLinkedDeque<>();

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
     * Track any AutoCloseable native resource (OnnxTensor, OrtSession.Result, etc.)
     * as a safety net. The resource will be closed when {@link #close()} is called,
     * even if the caller forgets to close it explicitly.
     */
    public <T extends AutoCloseable> T track(T closeable) {
        if (closeable != null) {
            closeables.push(closeable);
        }
        return closeable;
    }

    /**
     * Release a tracked AutoCloseable early (idempotent).
     */
    public void release(AutoCloseable closeable) {
        if (closeable != null && closeables.remove(closeable)) {
            IOUtil.close(closeable);
        }
    }

    /**
     * 当前被跟踪（尚未释放）的 Mat 数量。
     * 用于内存泄露验证：长期存活的 MatManager 在每轮推理后该值应回归到基线。
     */
    public int trackedMatCount() {
        return resources.size();
    }

    /**
     * 当前被跟踪（尚未释放）的 AutoCloseable 数量（OnnxTensor、OrtSession.Result 等）。
     * 用于内存泄露验证：每轮推理后该值应回归到 0。
     */
    public int trackedCloseableCount() {
        return closeables.size();
    }

    /**
     * Create an OnnxTensor from a FloatBuffer AND track it for automatic cleanup.
     */
    public OnnxTensor createOnnxTensor(OrtEnvironment env, FloatBuffer data, long[] shape) {
        try {
            return track(OnnxTensor.createTensor(env, data, shape));
        } catch (OrtException e) {
            throw new FluxException(e);
        }
    }

    /**
     * Create an OnnxTensor from a ByteBuffer AND track it for automatic cleanup.
     */
    public OnnxTensor createOnnxTensor(OrtEnvironment env, ByteBuffer data, long[] shape) {
        try {
            return track(OnnxTensor.createTensor(env, data, shape));
        } catch (OrtException e) {
            throw new FluxException(e);
        }
    }

    /**
     * Create an OnnxTensor from a LongBuffer AND track it for automatic cleanup.
     */
    public OnnxTensor createOnnxTensor(OrtEnvironment env, LongBuffer data, long[] shape) {
        try {
            return track(OnnxTensor.createTensor(env, data, shape));
        } catch (OrtException e) {
            throw new FluxException(e);
        }
    }

    /**
     * Run an ONNX session AND track the Result for automatic cleanup.
     */
    public OrtSession.Result runSession(OrtSession session, Map<String, OnnxTensor> inputs) {
        try {
            return track(session.run(inputs));
        } catch (OrtException e) {
            throw new FluxException(e);
        }
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
        AutoCloseable c;
        while ((c = closeables.poll()) != null) {
            IOUtil.close(c);
        }
        resources.forEach((_, mat) -> IOUtil.close(mat));
        resources.clear();
    }

}
