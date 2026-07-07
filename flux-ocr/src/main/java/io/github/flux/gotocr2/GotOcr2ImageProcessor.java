package io.github.flux.gotocr2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.MatManager;
import io.github.flux.formula.pix2text.DeiTImageProcessor;
import io.github.flux.paddle.processor.ResizeNdArray;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;

public class GotOcr2ImageProcessor {

    public static NDArray process(Mat rgbMat, MatManager matManager, NDManager ndManager) {
        NDArray image = ImageUtil.rgbToNDArray(rgbMat, ndManager);
        NDArray resized = new ResizeNdArray(1024, 1024, 2).process(matManager, List.of(image)).getFirst();
        NDArray rescaled = DeiTImageProcessor.rescale(resized, 0.00392156862745098f);
        resized.close();
        NDArray normalized = DeiTImageProcessor.normalize(rescaled,
                new float[]{0.48145466f, 0.4578275f, 0.40821073f},
                new float[]{0.26862954f, 0.26130258f, 0.27577711f}, ndManager);
        rescaled.close();
        // transpose 返回的是 normalized 的视图，其底层 base(normalized) 在视图关闭时
        // 不会自动释放，会残留于 NDManager（内存泄露）。duplicate 出独立副本后关闭
        // 视图与 base，确保仅返回 1 个独立 NDArray。输入 Mat 在此释放。
        NDArray transposed = normalized.transpose(2, 0, 1);
        NDArray resultNd = transposed.duplicate();
        transposed.close();
        normalized.close();
        matManager.release(rgbMat);
        return resultNd;
    }

}
