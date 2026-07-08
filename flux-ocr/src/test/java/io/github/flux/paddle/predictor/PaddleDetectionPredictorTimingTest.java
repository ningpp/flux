package io.github.flux.paddle.predictor;

import io.github.flux.core.MatManager;
import io.github.flux.paddle.processor.DBPostProcess;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PaddleDetectionPredictorTimingTest {

    @Test
    void detectionPredictorExposesNanosecondStageTimings() throws Exception {
        Class<?> timingsClass = Class.forName(PaddleDetectionPredictor.class.getName() + "$StageTimings");
        List<String> componentNames = Arrays.stream(timingsClass.getRecordComponents())
                .map(component -> component.getName())
                .toList();

        assertEquals(List.of(
                "padNanos",
                "resizeNanos",
                "preprocessNanos",
                "tensorCreateNanos",
                "inferenceNanos",
                "outputReadNanos",
                "postprocessNanos",
                "cleanupNanos",
                "totalNanos"), componentNames);

        Method timedMethod = PaddleDetectionPredictor.class.getMethod(
                "batchPredictWithTimings",
                List.class,
                MatManager.class,
                ai.djl.ndarray.NDManager.class,
                Integer.class,
                io.github.flux.paddle.processor.LimitType.class,
                Integer.class,
                Float.class,
                Float.class,
                Float.class);

        assertEquals("TimedResult", timedMethod.getReturnType().getSimpleName());
    }

    @Test
    void dbPostProcessCanProcessQuadBitmapFromFlatPredictionWithoutNdArrayBoxes() throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        DBPostProcess postProcess = new DBPostProcess(0.3f, 0.1f, 1.5f, 1000, "fast", "quad");
        Method method = DBPostProcess.class.getMethod(
                "boxesFromBitmap",
                MatManager.class,
                float[].class,
                int.class,
                int.class,
                int.class,
                int.class,
                float.class,
                float.class,
                float.class);

        float[] pred = new float[32 * 32];
        for (int y = 4; y <= 11; y++) {
            for (int x = 4; x <= 11; x++) {
                pred[y * 32 + x] = 0.95f;
            }
        }

        try (MatManager matManager = new MatManager()) {
            Object result = method.invoke(postProcess, matManager, pred, 32, 32, 32, 32, 0.3f, 0.1f, 1.5f);
            Method polysAccessor = result.getClass().getMethod("polys");
            Method scoresAccessor = result.getClass().getMethod("scores");
            int[][][] polys = (int[][][]) polysAccessor.invoke(result);
            @SuppressWarnings("unchecked")
            List<Float> scores = (List<Float>) scoresAccessor.invoke(result);

            assertEquals(1, polys.length);
            assertEquals(1, scores.size());
            assertTrue(scores.getFirst() > 0.9f);
            assertFalse(Arrays.deepEquals(new int[1][][], polys));
            assertEquals(0, matManager.trackedMatCount());
            assertEquals(0, matManager.trackedCloseableCount());
        }
    }
}
