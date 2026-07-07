package io.github.flux.paddle.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.MatManager;
import io.github.flux.util.ArrayUtil;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DBPostProcessMemoryTest {

    @BeforeAll
    static void loadOpenCv() {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    @Test
    void callKeepsPredictionAliveAcrossMultipleContours() throws Exception {
        try (MatManager matManager = new MatManager();
             NDManager manager = NDManager.newBaseManager()) {
            float[][][] data = new float[1][32][32];
            fillRect(data[0], 4, 4, 11, 11, 0.95f);
            fillRect(data[0], 20, 20, 27, 27, 0.95f);

            NDList preds = new NDList(ArrayUtil.toNDArray(manager, data));
            DBPostProcess postProcess = new DBPostProcess(0.3f, 0.1f, 1.5f, 1000, "fast", "quad");

            Pair<List<NDArray>, List<Float>> result = postProcess.call(
                    matManager,
                    preds,
                    new double[]{32, 32, 1, 1},
                    null,
                    null,
                    null);

            try {
                assertEquals(2, result.getKey().size());
                assertEquals(2, result.getValue().size());
                assertEquals(0.95f, preds.get(0).getFloat(0, 4, 4), 0.001f);
            } finally {
                for (NDArray box : result.getKey()) {
                    box.close();
                }
                preds.close();
            }
        }
    }

    private static void fillRect(float[][] target, int x0, int y0, int x1, int y1, float value) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                target[y][x] = value;
            }
        }
    }
}
