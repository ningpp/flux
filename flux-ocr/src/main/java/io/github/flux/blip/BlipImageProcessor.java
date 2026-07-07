package io.github.flux.blip;

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
     * 将 RGB Mat 处理为归一化、形状 [C,H,W] 的 float[]（channel 顺序 R,G,B）。
     *
     * <p>直接返回 {@code float[]} 而非 DJL {@code NDArray}：BLIP 仅需要这份像素数据喂给
     * ONNX vision encoder，绕开 “float[] -> NDArray -> float[]” 的冗余原生内存分配与来回拷贝，
     * 既降低内存峰值，也消除调用方若不及时 close 而累积的 NDArray 原生内存泄漏。</p>
     *
     * <p>处理过程中的中间 Mat（resized / floatMat）在读取完像素后立即释放，
     * 避免 MatManager 跟踪表随推理张数无限增长导致原生内存泄漏。</p>
     */
    public static float[] process(Mat rgbMat, MatManager matManager) {
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

        // 中间 Mat 已使用完毕，立即释放，避免 MatManager 跟踪表无限累积（内存泄露）
        matManager.release(resized);
        matManager.release(floatMat);

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

        return chwData;
    }
}
