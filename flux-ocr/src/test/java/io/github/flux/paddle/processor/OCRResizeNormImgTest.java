package io.github.flux.paddle.processor;

import io.github.flux.core.MatManager;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class OCRResizeNormImgTest {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    @Test
    void normalizeToChwKeepsPaddingZeroAndChannelOrder() throws Exception {
        try (MatManager matManager = new MatManager()) {
            Mat resized = matManager.newMat(1, 2, CvType.CV_8UC3);
            resized.put(0, 0, new byte[]{
                    0, 127, (byte) 255,
                    10, 20, 30
            });

            float[] actual = OCRResizeNormImg.normalizeToChw(resized, 3, 1, 4, 2);

            float[] expected = {
                    normalize(0), normalize(10), 0.0f, 0.0f,
                    normalize(127), normalize(20), 0.0f, 0.0f,
                    normalize(255), normalize(30), 0.0f, 0.0f
            };
            assertArrayEquals(expected, actual, 1e-6f);
        }
    }

    private static float normalize(int value) {
        return value * (2.0f / 255.0f) - 1.0f;
    }
}
