package io.github.flux.formula.pix2text;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class Pix2TextPreProcessor {

    public NDArray process(Mat rgbImg, NDManager manager) {
        DeiTImageProcessor deitProcessor = DeiTImageProcessor.builder()
                .setDoResize(true)
                .setSize(384, 384)
                .setDoCenterCrop(false)
                .setCropSize(224, 224)
                .setDoRescale(true)
                .setRescaleFactor(1.0f / 255.0f)
                .setDoNormalize(true)
                .setImageMean(new float[]{0.5f, 0.5f, 0.5f})
                .setImageStd(new float[]{0.5f, 0.5f, 0.5f})
                // 在 Pillow（PIL）中，Image.BICUBIC 表示三次卷积插值（Bicubic interpolation），其值为 Image.BICUBIC = 3
                // 在 OpenCV 中，对应的是：cv2.INTER_CUBIC  # 值为 2
                .setResampleMode(Imgproc.INTER_CUBIC)
                .build();

        NDArray imgNdArray = ImageUtil.toNDArrayUint8(rgbImg, manager);
        NDArray input = deitProcessor.preprocess(imgNdArray, manager);

        rgbImg.release();
        imgNdArray.close();
        return input;
    }

}
