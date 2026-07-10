package io.github.flux.pipeline;

import ai.djl.ndarray.NDManager;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import io.github.flux.core.TextDetectionResult;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

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
        writeImage(imagePath);

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

    @Test
    void predictBatchesLayoutRegionTextAndFormulaWorkAcrossImages() throws Exception {
        Path firstImage = tempDir.resolve("page-1.png");
        Path secondImage = tempDir.resolve("page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        FakeClassificationModel docOrientationModel = new FakeClassificationModel("0");
        FakeLayoutModel layoutModel = new FakeLayoutModel();
        FakeFormulaModel formulaModel = new FakeFormulaModel();
        FakeTextDetectionModel textDetectionModel = new FakeTextDetectionModel();
        FakeClassificationModel textLineOrientationModel = new FakeClassificationModel("0_degree");
        FakeRecognitionModel recognitionModel = new FakeRecognitionModel();

        OCRPipeline pipeline = new OCRPipeline(
                textDetectionModel,
                recognitionModel,
                docOrientationModel,
                textLineOrientationModel,
                layoutModel,
                formulaModel,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of(
                            "layoutBatchSize", 2,
                            "detectionBatchSize", 2,
                            "recognitionBatchSize", 8,
                            "formulaBatchSize", 8),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(2, docOrientationModel.maxBatchSize);
            assertEquals(2, layoutModel.maxBatchSize);
            assertEquals(2, formulaModel.maxBatchSize);
            assertEquals(2, textDetectionModel.maxBatchSize);
            assertEquals(2, textLineOrientationModel.maxBatchSize);
            assertEquals(2, recognitionModel.maxBatchSize);
            assertEquals(1, formulaModel.callCount);
            assertEquals(1, textDetectionModel.callCount);
            assertEquals(1, textLineOrientationModel.callCount);
            assertEquals(1, recognitionModel.callCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictDefaultsLimitLayoutAndDetectionBatchesIndependentlyFromRecognitionBatchSize() throws Exception {
        Path firstImage = tempDir.resolve("safe-default-page-1.png");
        Path secondImage = tempDir.resolve("safe-default-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        FakeLayoutModel layoutModel = new FakeLayoutModel();
        FakeTextDetectionModel textDetectionModel = new FakeTextDetectionModel();
        FakeRecognitionModel recognitionModel = new FakeRecognitionModel();

        OCRPipeline pipeline = new OCRPipeline(
                textDetectionModel,
                recognitionModel,
                null,
                null,
                layoutModel,
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("recognitionBatchSize", 8),
                    matManager,
                    ndManager);

            assertEquals(1, layoutModel.maxBatchSize);
            assertEquals(1, textDetectionModel.maxBatchSize);
            assertEquals(1, recognitionModel.maxBatchSize);
        }
    }

    @Test
    void predictRetriesFormulaBatchWithSmallerBatchesWhenLargeBatchFails() throws Exception {
        Path firstImage = tempDir.resolve("formula-page-1.png");
        Path secondImage = tempDir.resolve("formula-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        RetryingFormulaModel formulaModel = new RetryingFormulaModel();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                null,
                null,
                new FormulaOnlyLayoutModel(),
                formulaModel,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("formulaBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(1, formulaModel.failedLargeBatchCount);
            assertEquals(2, formulaModel.singleBatchCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictRetriesTableBatchWithSmallerBatchesWhenLargeBatchFails() throws Exception {
        Path firstImage = tempDir.resolve("table-page-1.png");
        Path secondImage = tempDir.resolve("table-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        RetryingTableModel tableModel = new RetryingTableModel();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                null,
                null,
                new TableOnlyLayoutModel(),
                null,
                tableModel);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("tableBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(1, tableModel.failedLargeBatchCount);
            assertEquals(2, tableModel.singleBatchCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictRetriesTextDetectionBatchWithSmallerBatchesWhenLargeBatchFails() throws Exception {
        Path firstImage = tempDir.resolve("text-page-1.png");
        Path secondImage = tempDir.resolve("text-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        RetryingTextDetectionModel textDetectionModel = new RetryingTextDetectionModel();
        OCRPipeline pipeline = new OCRPipeline(
                textDetectionModel,
                new FakeRecognitionModel(),
                null,
                null,
                new TextOnlyLayoutModel(),
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("layoutBatchSize", 2, "detectionBatchSize", 2, "recognitionBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(1, textDetectionModel.failedLargeBatchCount);
            assertEquals(2, textDetectionModel.singleBatchCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictRetriesLayoutBatchWithSmallerBatchesWhenLargeBatchFails() throws Exception {
        Path firstImage = tempDir.resolve("layout-page-1.png");
        Path secondImage = tempDir.resolve("layout-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        RetryingLayoutModel layoutModel = new RetryingLayoutModel();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                null,
                null,
                layoutModel,
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("layoutBatchSize", 2, "detectionBatchSize", 2, "recognitionBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(1, layoutModel.failedLargeBatchCount);
            assertEquals(2, layoutModel.singleBatchCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictRecreatesLayoutInputsBeforeRetryWhenFailedAttemptConsumesThem() throws Exception {
        Path firstImage = tempDir.resolve("consumed-layout-page-1.png");
        Path secondImage = tempDir.resolve("consumed-layout-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        ConsumingRetryingLayoutModel layoutModel = new ConsumingRetryingLayoutModel();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                null,
                null,
                layoutModel,
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predict(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("detectionBatchSize", 2, "recognitionBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(1, layoutModel.failedLargeBatchCount);
            assertEquals(2, layoutModel.singleBatchCount);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictReleasesTextLineCropsBetweenRecognitionBatches() throws Exception {
        Path imagePath = tempDir.resolve("many-lines-page.png");
        writeImage(imagePath);

        TrackingRecognitionModel recognitionModel = new TrackingRecognitionModel();
        OCRPipeline pipeline = new OCRPipeline(
                new MultiLineTextDetectionModel(),
                recognitionModel,
                null,
                null,
                null,
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            pipeline.predict(
                    List.of(imagePath.toString()),
                    Map.of("recognitionBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(List.of(2, 2), recognitionModel.batchSizes);
            assertEquals(2, recognitionModel.trackedMatCounts.get(0) - recognitionModel.trackedMatCounts.get(1));
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictEmitsMemoryObserverStagesWithoutChangingPipelineOwnership() throws Exception {
        Path imagePath = tempDir.resolve("observed-page.png");
        writeImage(imagePath);

        List<String> stages = new ArrayList<>();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                new FakeClassificationModel("0"),
                new FakeClassificationModel("0_degree"),
                new FakeLayoutModel(),
                new FakeFormulaModel(),
                new FakeTableModel());

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            pipeline.predict(
                    List.of(imagePath.toString()),
                    Map.of(
                            "recognitionBatchSize", 2,
                            "tableBatchSize", 1,
                            "memoryObserver", (java.util.function.Consumer<String>) stages::add),
                    matManager,
                    ndManager);

            assertStageOrder(stages,
                    "predict:start",
                    "images:loaded",
                    "doc-orientation:done",
                    "layout:done",
                    "formula:done",
                    "table:done",
                    "text-detection:done",
                    "textline-orientation:done",
                    "text-recognition:done",
                    "text:done",
                    "predict:released");
        }
    }

    private void writeImage(Path imagePath) throws Exception {
        try (MatManager matManager = new MatManager()) {
            Mat img = matManager.newMat(80, 120, CvType.CV_8UC3);
            Imgcodecs.imwrite(imagePath.toString(), img);
            matManager.release(img);
        }
    }

    private static void assertStageOrder(List<String> stages, String... expectedStages) {
        int previous = -1;
        for (String expectedStage : expectedStages) {
            int index = stages.indexOf(expectedStage);
            assertFalse(index < 0, "missing stage " + expectedStage + " in " + stages);
            assertFalse(index <= previous, "stage order is wrong: " + stages);
            previous = index;
        }
    }

    private static class FakeLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {
        int callCount;
        int maxBatchSize;

        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(0, "text", 0.9f, new float[]{0, 0, 100, 70}),
                            new ObjectDetectionResult(1, "display_formula", 0.9f, new float[]{0, 0, 100, 70})))
                    .toList();
        }

        @Override
        public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
        }

        @Override
        public void close() {
        }
    }

    private static class FormulaOnlyLayoutModel extends FakeLayoutModel {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(1, "display_formula", 0.9f, new float[]{0, 0, 100, 70})))
                    .toList();
        }
    }

    private static class TextOnlyLayoutModel extends FakeLayoutModel {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(0, "text", 0.9f, new float[]{0, 0, 100, 70})))
                    .toList();
        }
    }

    private static class TableOnlyLayoutModel extends FakeLayoutModel {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(2, "table", 0.9f, new float[]{0, 0, 100, 70})))
                    .toList();
        }
    }

    private static class RetryingLayoutModel extends TextOnlyLayoutModel {
        int failedLargeBatchCount;
        int singleBatchCount;

        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            if (mats.size() > 1) {
                failedLargeBatchCount++;
                throw new RuntimeException("simulated layout batch OOM");
            }
            singleBatchCount++;
            return super.doBatchPredict(mats, matManager, ndManager, extraParameters);
        }
    }

    private static class ConsumingRetryingLayoutModel extends RetryingLayoutModel {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            if (mats.size() > 1) {
                failedLargeBatchCount++;
                for (ProcessedMat mat : mats) {
                    matManager.release(mat.processed());
                }
                throw new RuntimeException("simulated layout batch OOM after preprocessing consumed inputs");
            }
            for (ProcessedMat mat : mats) {
                assertFalse(mat.processed().empty(), "layout retry must use freshly created inputs");
            }
            singleBatchCount++;
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(0, "text", 0.9f, new float[]{0, 0, 100, 70})))
                    .toList();
        }
    }

    private static class FakeTextDetectionModel extends BatchPredictor<PreProcessResult, TextDetectionResult> {
        int callCount;
        int maxBatchSize;

        @Override
        public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                        NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> {
                        int[][][] polys = new int[][][]{
                                {{5, 5}, {55, 5}, {55, 25}, {5, 25}}
                        };
                        return new TextDetectionResult(polys, List.of(0.95f));
                    })
                    .toList();
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
        int callCount;
        int maxBatchSize;

        @Override
        public List<List<RecognitionResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                            NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(new RecognitionResult("ok", new double[]{1.0})))
                    .toList();
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }

    private static class MultiLineTextDetectionModel extends FakeTextDetectionModel {
        @Override
        public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                        NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> {
                        int[][][] polys = new int[][][]{
                                {{5, 5}, {55, 5}, {55, 15}, {5, 15}},
                                {{5, 20}, {55, 20}, {55, 30}, {5, 30}},
                                {{5, 35}, {55, 35}, {55, 45}, {5, 45}},
                                {{5, 50}, {55, 50}, {55, 60}, {5, 60}}
                        };
                        return new TextDetectionResult(polys, List.of(0.95f, 0.95f, 0.95f, 0.95f));
                    })
                    .toList();
        }
    }

    private static class TrackingRecognitionModel extends FakeRecognitionModel {
        final List<Integer> trackedMatCounts = new ArrayList<>();
        final List<Integer> batchSizes = new ArrayList<>();

        @Override
        public List<List<RecognitionResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                            NDManager ndManager, Map<String, Object> extraParameters) {
            trackedMatCounts.add(matManager.trackedMatCount());
            batchSizes.add(mats.size());
            return super.doBatchPredict(mats, matManager, ndManager, extraParameters);
        }
    }

    private static class FakeClassificationModel extends BatchPredictor<PreProcessResult, ClassificationResult> {
        private final String label;
        int callCount;
        int maxBatchSize;

        FakeClassificationModel(String label) {
            this.label = label;
        }

        @Override
        public List<ClassificationResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                         NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return IntStream.range(0, mats.size())
                    .mapToObj(_ -> new ClassificationResult(0.99f, label))
                    .toList();
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }

    private static class FakeFormulaModel extends BatchPredictor<PreProcessResult, TextResult> {
        int callCount;
        int maxBatchSize;

        @Override
        public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                               NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> new TextResult("x", new long[]{1L}, 0.99f))
                    .toList();
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }

    private static class FakeTableModel extends BatchPredictor<PreProcessResult, TableResult> {
        int callCount;
        int maxBatchSize;

        @Override
        public List<TableResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> new TableResult("table", new long[]{1L}))
                    .toList();
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() {
        }
    }

    private static class RetryingFormulaModel extends FakeFormulaModel {
        int failedLargeBatchCount;
        int singleBatchCount;

        @Override
        public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                               NDManager ndManager, Map<String, Object> extraParameters) {
            if (mats.size() > 1) {
                failedLargeBatchCount++;
                throw new RuntimeException("simulated formula batch OOM");
            }
            singleBatchCount++;
            return super.doBatchPredict(mats, matManager, ndManager, extraParameters);
        }
    }

    private static class RetryingTableModel extends FakeTableModel {
        int failedLargeBatchCount;
        int singleBatchCount;

        @Override
        public List<TableResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                NDManager ndManager, Map<String, Object> extraParameters) {
            if (mats.size() > 1) {
                failedLargeBatchCount++;
                throw new RuntimeException("simulated table batch OOM");
            }
            singleBatchCount++;
            return super.doBatchPredict(mats, matManager, ndManager, extraParameters);
        }
    }

    private static class RetryingTextDetectionModel extends FakeTextDetectionModel {
        int failedLargeBatchCount;
        int singleBatchCount;

        @Override
        public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                        NDManager ndManager, Map<String, Object> extraParameters) {
            if (mats.size() > 1) {
                failedLargeBatchCount++;
                throw new RuntimeException("simulated text detection batch OOM");
            }
            singleBatchCount++;
            return super.doBatchPredict(mats, matManager, ndManager, extraParameters);
        }
    }

    private static class MixedRegionsLayoutModel extends FakeLayoutModel {
        @Override
        public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                                NDManager ndManager, Map<String, Object> extraParameters) {
            callCount++;
            maxBatchSize = Math.max(maxBatchSize, mats.size());
            return mats.stream()
                    .map(_ -> List.of(
                            new ObjectDetectionResult(0, "text", 0.9f, new float[]{0, 0, 100, 30}),
                            new ObjectDetectionResult(1, "display_formula", 0.9f, new float[]{0, 30, 100, 60}),
                            new ObjectDetectionResult(2, "table", 0.9f, new float[]{0, 60, 100, 90})))
                    .toList();
        }
    }

    @Test
    void predictV2BatchesRecognitionAcrossImagesAndDetectionBatches() throws Exception {
        Path firstImage = tempDir.resolve("v2-multi-line-page-1.png");
        Path secondImage = tempDir.resolve("v2-multi-line-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        FakeRecognitionModel recognitionModel = new FakeRecognitionModel();
        OCRPipeline pipeline = new OCRPipeline(
                new MultiLineTextDetectionModel(),
                recognitionModel,
                null,
                null,
                null,
                null,
                null);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predictV2(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of("detectionBatchSize", 1, "recognitionBatchSize", 3),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            // 2 pages × 4 lines/page = 8 lines, batch 3 → 3 batches (3+3+2)
            assertEquals(3, recognitionModel.callCount);
            assertEquals(3, recognitionModel.maxBatchSize);
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictV2ReleasesAllTrackedMats() throws Exception {
        Path firstImage = tempDir.resolve("v2-mixed-page-1.png");
        Path secondImage = tempDir.resolve("v2-mixed-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                new FakeClassificationModel("0"),
                new FakeClassificationModel("0_degree"),
                new MixedRegionsLayoutModel(),
                new FakeFormulaModel(),
                new FakeTableModel());

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            int baseline = matManager.trackedMatCount();
            int closeableBaseline = matManager.trackedCloseableCount();

            List<List<OCRPipelineResult>> results = pipeline.predictV2(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of(
                            "layoutBatchSize", 2,
                            "detectionBatchSize", 2,
                            "recognitionBatchSize", 4,
                            "formulaBatchSize", 2,
                            "tableBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, results.size());
            assertEquals(baseline, matManager.trackedMatCount());
            assertEquals(closeableBaseline, matManager.trackedCloseableCount());
        }
    }

    @Test
    void predictV2IndependentBatchSizesForAllModels() throws Exception {
        Path firstImage = tempDir.resolve("v2-batch-page-1.png");
        Path secondImage = tempDir.resolve("v2-batch-page-2.png");
        writeImage(firstImage);
        writeImage(secondImage);

        FakeClassificationModel docOrientationModel = new FakeClassificationModel("0");
        MixedRegionsLayoutModel layoutModel = new MixedRegionsLayoutModel();
        FakeTextDetectionModel textDetectionModel = new FakeTextDetectionModel();
        FakeClassificationModel textLineOrientationModel = new FakeClassificationModel("0_degree");
        FakeRecognitionModel recognitionModel = new FakeRecognitionModel();
        FakeFormulaModel formulaModel = new FakeFormulaModel();
        FakeTableModel tableModel = new FakeTableModel();

        OCRPipeline pipeline = new OCRPipeline(
                textDetectionModel,
                recognitionModel,
                docOrientationModel,
                textLineOrientationModel,
                layoutModel,
                formulaModel,
                tableModel);

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            pipeline.predictV2(
                    List.of(firstImage.toString(), secondImage.toString()),
                    Map.of(
                            "layoutBatchSize", 2,
                            "docOrientationBatchSize", 1,
                            "textLineOrientationBatchSize", 1,
                            "detectionBatchSize", 1,
                            "recognitionBatchSize", 2,
                            "formulaBatchSize", 1,
                            "tableBatchSize", 2),
                    matManager,
                    ndManager);

            assertEquals(2, layoutModel.maxBatchSize);
            assertEquals(1, docOrientationModel.maxBatchSize);
            assertEquals(1, textLineOrientationModel.maxBatchSize);
            assertEquals(1, textDetectionModel.maxBatchSize);
            assertEquals(2, recognitionModel.maxBatchSize);
            assertEquals(1, formulaModel.maxBatchSize);
            assertEquals(2, tableModel.maxBatchSize);
        }
    }

    @Test
    void predictV2EmitsExpectedStageOrder() throws Exception {
        Path imagePath = tempDir.resolve("v2-observed-page.png");
        writeImage(imagePath);

        List<String> stages = new ArrayList<>();
        OCRPipeline pipeline = new OCRPipeline(
                new FakeTextDetectionModel(),
                new FakeRecognitionModel(),
                new FakeClassificationModel("0"),
                new FakeClassificationModel("0_degree"),
                new MixedRegionsLayoutModel(),
                new FakeFormulaModel(),
                new FakeTableModel());

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            pipeline.predictV2(
                    List.of(imagePath.toString()),
                    Map.of(
                            "recognitionBatchSize", 2,
                            "tableBatchSize", 1,
                            "memoryObserver", (java.util.function.Consumer<String>) stages::add),
                    matManager,
                    ndManager);

            assertStageOrder(stages,
                    "predictV2:start",
                    "images:loaded",
                    "doc-orientation:done",
                    "layout:done",
                    "formula:done",
                    "table:done",
                    "text-detection:done",
                    "textline-orientation:done",
                    "text-recognition:done",
                    "text:done",
                    "predictV2:released");
        }
    }

}
