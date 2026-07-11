package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.paddle.predictor.PaddleRecognitionPredictor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Verify det+rec behavior on a single image, focusing on pad-to-same impact.
 *
 * Usage:
 * mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.DetRecPadVerify" "-Dexec.classpathScope=test" "-Dexec.args=E:\\flux-data\\text-20260711104047.png 0"
 */
public class DetRecPadVerify {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String OUTPUT_LOG = "E:\\flux-data\\det-rec-pad-verify.log";

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String imagePath = args.length > 0 ? args[0] : "E:\\flux-data\\text-20260711104047.png";
        int gpuIndex = args.length > 1 ? Integer.parseInt(args[1]) : 0;
        int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 3;
        int loops = args.length > 3 ? Integer.parseInt(args[3]) : 10;

        try (PrintStream logOut = new PrintStream(new FileOutputStream(OUTPUT_LOG, true), true, StandardCharsets.UTF_8)) {
            TeePrintStream tee = new TeePrintStream(System.out, logOut);
            runVerify(tee, imagePath, gpuIndex, warmup, loops);
        }
    }

    private static void runVerify(PrintStream out, String imagePath, int gpuIndex, int warmup, int loops) throws Exception {
        out.println("=================================================================");
        out.println("DetRecPadVerify start at " + LocalDateTime.now());
        out.printf(Locale.ROOT, "image=%s gpuIndex=%d warmup=%d loops=%d%n", imagePath, gpuIndex, warmup, loops);

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            // 1. Read image as BGR
            Mat bgrImage = matManager.imread(imagePath, Imgcodecs.IMREAD_COLOR_BGR);
            if (bgrImage == null || bgrImage.empty()) {
                throw new IllegalArgumentException("Failed to read image: " + imagePath);
            }
            out.printf(Locale.ROOT, "input image size=%dx%d channels=%d%n", bgrImage.cols(), bgrImage.rows(), bgrImage.channels());

            // 2. Convert to RGB for detection
            Mat rgbImage = matManager.newMat();
            Imgproc.cvtColor(bgrImage, rgbImage, Imgproc.COLOR_BGR2RGB);

            // 3. Text detection
            long detStart = System.nanoTime();
            List<TextDetectionResult> detResults = detModel.batchPredict(
                    List.of(new PreProcessResult(rgbImage, null)),
                    1, matManager, ndManager, Map.of());
            long detNanos = System.nanoTime() - detStart;
            TextDetectionResult detResult = detResults.getFirst();
            int lineCount = detResult.polys() == null ? 0 : detResult.polys().length;
            out.printf(Locale.ROOT, "detection lines=%d timeMs=%.2f%n", lineCount, detNanos / 1_000_000d);

            if (lineCount == 0) {
                out.println("No text lines detected.");
                return;
            }

            // 4. Crop text lines (from BGR image, then convert each to RGB for rec)
            List<LineCrop> lineCrops = new ArrayList<>();
            for (int i = 0; i < lineCount; i++) {
                Mat lineBgr = ImageUtil.getMinAreaRectCrop(matManager, ndManager, bgrImage, detResult.polys()[i]);
                Mat lineRgb = matManager.newMat();
                Imgproc.cvtColor(lineBgr, lineRgb, Imgproc.COLOR_BGR2RGB);
                matManager.release(lineBgr);
                lineCrops.add(new LineCrop(i, lineRgb));
            }
            matManager.release(bgrImage);
            matManager.release(rgbImage);

            // Print crop dimensions
            out.println("\n--- Line crop dimensions ---");
            for (LineCrop lc : lineCrops) {
                out.printf(Locale.ROOT, "line=%02d width=%4d height=%4d ratio=%.3f%n",
                        lc.index, lc.rgb.cols(), lc.rgb.rows(), lc.rgb.cols() / (double) lc.rgb.rows());
            }

            // Summary stats
            int minW = lineCrops.stream().mapToInt(c -> c.rgb.cols()).min().orElse(0);
            int maxW = lineCrops.stream().mapToInt(c -> c.rgb.cols()).max().orElse(0);
            int avgW = (int) lineCrops.stream().mapToInt(c -> c.rgb.cols()).average().orElse(0);
            int sumW = lineCrops.stream().mapToInt(c -> c.rgb.cols()).sum();
            out.printf(Locale.ROOT, "\ncrop width stats: min=%d max=%d avg=%d sum=%d%n", minW, maxW, avgW, sumW);
            out.printf(Locale.ROOT, "theoretical batch-1 total width=%d, batch-all tensor width=%d (overhead %.2fx)%n",
                    sumW, maxW * lineCrops.size(), (maxW * (double) lineCrops.size()) / Math.max(1, sumW));

            PaddleRecognitionPredictor recPredictor = recModel.getPredictor();

            // Warmup: clone each line before rec, because current predictor may release input Mat when no padding
            for (int i = 0; i < warmup; i++) {
                for (LineCrop lc : lineCrops) {
                    Mat clone = matManager.cloneMat(lc.rgb);
                    recPredictor.batchPredictWithTimings(List.of(clone), matManager, ndManager);
                }
            }

            // 5. Recognition batch=1
            out.println("\n--- Recognition batch=1 (loop over all lines) ---");
            long recBatch1Total = 0;
            long recBatch1Pad = 0;
            long recBatch1Pre = 0;
            long recBatch1Inf = 0;
            List<String> textsBatch1 = new ArrayList<>();
            for (int loop = 0; loop < loops; loop++) {
                for (LineCrop lc : lineCrops) {
                    Mat clone = matManager.cloneMat(lc.rgb);
                    long t0 = System.nanoTime();
                    PaddleRecognitionPredictor.TimedResult tr =
                            recPredictor.batchPredictWithTimings(List.of(clone), matManager, ndManager);
                    long e2e = System.nanoTime() - t0;
                    PaddleRecognitionPredictor.StageTimings t = tr.timings();
                    recBatch1Total += e2e;
                    recBatch1Pad += t.padNanos();
                    recBatch1Pre += t.preprocessNanos();
                    recBatch1Inf += t.inferenceNanos();
                    if (loop == 0) {
                        textsBatch1.add(tr.results().getFirst().getFirst().text());
                    }
                }
            }
            out.printf(Locale.ROOT, "batch=1 avg per-line total=%.3fms pad=%.3fms preprocess=%.3fms inference=%.3fms%n",
                    (recBatch1Total / (double) loops / lineCount) / 1_000_000d,
                    (recBatch1Pad / (double) loops / lineCount) / 1_000_000d,
                    (recBatch1Pre / (double) loops / lineCount) / 1_000_000d,
                    (recBatch1Inf / (double) loops / lineCount) / 1_000_000d);
            out.println("batch=1 texts:");
            for (int i = 0; i < textsBatch1.size(); i++) {
                out.printf(Locale.ROOT, "  [%02d] %s%n", i, abbreviate(textsBatch1.get(i)));
            }

            // 6. Recognition batch=all (current behavior: padImageToSame before resize)
            out.println("\n--- Recognition batch=all (current pad-before-resize) ---");
            long recBatchAllTotal = 0;
            long recBatchAllPad = 0;
            long recBatchAllPre = 0;
            long recBatchAllInf = 0;
            List<String> textsBatchAll = new ArrayList<>();
            for (int loop = 0; loop < loops; loop++) {
                List<Mat> allLines = new ArrayList<>(lineCrops.size());
                for (LineCrop lc : lineCrops) {
                    allLines.add(matManager.cloneMat(lc.rgb));
                }
                long t0 = System.nanoTime();
                PaddleRecognitionPredictor.TimedResult tr =
                        recPredictor.batchPredictWithTimings(allLines, matManager, ndManager);
                long e2e = System.nanoTime() - t0;
                PaddleRecognitionPredictor.StageTimings t = tr.timings();
                recBatchAllTotal += e2e;
                recBatchAllPad += t.padNanos();
                recBatchAllPre += t.preprocessNanos();
                recBatchAllInf += t.inferenceNanos();
                if (loop == 0) {
                    for (List<RecognitionResult> lr : tr.results()) {
                        textsBatchAll.add(lr.getFirst().text());
                    }
                }
            }
            out.printf(Locale.ROOT, "batch=all avg per-line total=%.3fms pad=%.3fms preprocess=%.3fms inference=%.3fms%n",
                    (recBatchAllTotal / (double) loops / lineCount) / 1_000_000d,
                    (recBatchAllPad / (double) loops / lineCount) / 1_000_000d,
                    (recBatchAllPre / (double) loops / lineCount) / 1_000_000d,
                    (recBatchAllInf / (double) loops / lineCount) / 1_000_000d);
            out.println("batch=all texts:");
            for (int i = 0; i < textsBatchAll.size(); i++) {
                out.printf(Locale.ROOT, "  [%02d] %s%n", i, abbreviate(textsBatchAll.get(i)));
            }

            // 7. Recognition batch=all sorted by width ratio (PaddleX style)
            out.println("\n--- Recognition batch=all sorted by ratio (PaddleX style) ---");
            List<LineCrop> sortedCrops = new ArrayList<>(lineCrops);
            sortedCrops.sort(Comparator.comparingDouble(c -> c.rgb.cols() / (double) c.rgb.rows()));
            long recBatchSortedTotal = 0;
            long recBatchSortedPad = 0;
            long recBatchSortedPre = 0;
            long recBatchSortedInf = 0;
            List<String> textsBatchSorted = new ArrayList<>();
            for (int loop = 0; loop < loops; loop++) {
                List<Mat> sortedLines = new ArrayList<>(sortedCrops.size());
                for (LineCrop lc : sortedCrops) {
                    sortedLines.add(matManager.cloneMat(lc.rgb));
                }
                long t0 = System.nanoTime();
                PaddleRecognitionPredictor.TimedResult tr =
                        recPredictor.batchPredictWithTimings(sortedLines, matManager, ndManager);
                long e2e = System.nanoTime() - t0;
                PaddleRecognitionPredictor.StageTimings t = tr.timings();
                recBatchSortedTotal += e2e;
                recBatchSortedPad += t.padNanos();
                recBatchSortedPre += t.preprocessNanos();
                recBatchSortedInf += t.inferenceNanos();
                if (loop == 0) {
                    for (List<RecognitionResult> lr : tr.results()) {
                        textsBatchSorted.add(lr.getFirst().text());
                    }
                }
            }
            out.printf(Locale.ROOT, "batch=all-sorted avg per-line total=%.3fms pad=%.3fms preprocess=%.3fms inference=%.3fms%n",
                    (recBatchSortedTotal / (double) loops / lineCount) / 1_000_000d,
                    (recBatchSortedPad / (double) loops / lineCount) / 1_000_000d,
                    (recBatchSortedPre / (double) loops / lineCount) / 1_000_000d,
                    (recBatchSortedInf / (double) loops / lineCount) / 1_000_000d);

            // 8. Summary comparison
            out.println("\n--- Summary ---");
            out.printf(Locale.ROOT, "batch=1      total=%.2fms per-page-equivalent%n",
                    (recBatch1Total / (double) loops) / 1_000_000d);
            out.printf(Locale.ROOT, "batch=all    total=%.2fms (%.2fx vs batch=1)%n",
                    (recBatchAllTotal / (double) loops) / 1_000_000d,
                    recBatchAllTotal / (double) recBatch1Total);
            out.printf(Locale.ROOT, "batch=sorted total=%.2fms (%.2fx vs batch=1)%n",
                    (recBatchSortedTotal / (double) loops) / 1_000_000d,
                    recBatchSortedTotal / (double) recBatch1Total);

            // Correctness check
            boolean match = textsBatch1.size() == textsBatchAll.size();
            if (match) {
                for (int i = 0; i < textsBatch1.size(); i++) {
                    if (!textsBatch1.get(i).equals(textsBatchAll.get(i))) {
                        match = false;
                        break;
                    }
                }
            }
            out.println("texts match between batch=1 and batch=all: " + match);

            // Cleanup
            for (LineCrop lc : lineCrops) {
                matManager.release(lc.rgb);
            }
            out.println("\nDetRecPadVerify end at " + LocalDateTime.now());
            out.println("=================================================================\n");
        }
    }

    private static String abbreviate(String text) {
        if (text == null) {
            return "";
        }
        String normalized = text.replace("\r", "\\r").replace("\n", "\\n");
        return normalized.length() <= 120 ? normalized : normalized.substring(0, 120) + "...";
    }

    private record LineCrop(int index, Mat rgb) {
    }

    private static class TeePrintStream extends PrintStream {
        private final PrintStream second;

        TeePrintStream(PrintStream first, PrintStream second) {
            super(first, true, StandardCharsets.UTF_8);
            this.second = second;
        }

        @Override
        public void write(int b) {
            super.write(b);
            second.write(b);
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            super.write(buf, off, len);
            second.write(buf, off, len);
        }

        @Override
        public void flush() {
            super.flush();
            second.flush();
        }
    }
}
