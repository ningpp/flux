package io.github.flux.pipeline;

import ai.djl.ndarray.NDManager;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TextDetectionResult;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OCRPipelineMemoryTest {

    @TempDir
    Path tempDir;

    @BeforeAll
    static void loadOpenCv() {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    @Test
    void predictReleasesTrackedMatsAfterLayoutTextPipeline() throws Exception {
        Path imagePath = tempDir.resolve("page.png");
        try (MatManager matManager = new MatManager()) {
            Mat img = matManager.newMat(80, 120, CvType.CV_8UC3);
            Imgcodecs.imwrite(imagePath.toString(), img);
            matManager.release(img);
        }

        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                null,
                null,
                new FakeLayoutModel(),
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            pipeline.predict(List.of(imagePath.toString()), Map.of("recognitionBatchSize", 2), matManager, ndManager);

            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    private static class FakeLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            return List.of(List.of(new ObjectDetectionResult(0, "text", 0.9f, new float[]{0, 0, 100, 70})));
        }

        @Override
        public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
        }

        @Override
        public void close() {
        }
    }

    private static class FakeTextDetectionModel extends BatchPredictor<PreProcessResult, TextDetectionResult> {
        @Override
        public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                        NDManager ndManager, Map<String, Object> extraParameters) {
            int[][][] polys = new int[][][]{
                    {{5, 5}, {55, 5}, {55, 25}, {5, 25}}
            };
            return List.of(new TextDetectionResult(polys, List.of(0.95f)));
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }

    private static class FakeRecognitionModel extends BatchPredictor<PreProcessResult, List<RecognitionResult>> {
        @Override
        public List<List<RecognitionResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                            NDManager ndManager, Map<String, Object> extraParameters) {
            assertEquals(1, mats.size());
            return List.of(List.of(new RecognitionResult("ok", new double[]{1.0})));
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }
}
