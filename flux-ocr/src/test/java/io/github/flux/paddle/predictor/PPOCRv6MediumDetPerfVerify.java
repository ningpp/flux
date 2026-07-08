package io.github.flux.paddle.predictor;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PPOCRv6MediumDetPerfVerify {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String MODEL_NAME = "PP-OCRv6_medium_det";
    private static final String DEFAULT_IMAGE = "D:\\tmp\\text-det-20260708204923.png";
    private static final Pattern RSS_PATTERN = Pattern.compile("([\\d,]+)\\s*K");

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String image = args.length > 0 ? args[0] : DEFAULT_IMAGE;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 30;
        int warmups = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        int gpuIndex = args.length > 3 ? Integer.parseInt(args[3]) : 0;
        long pid = ProcessHandle.current().pid();

        if (gpuIndex < 0) {
            throw new IllegalArgumentException("GPU inference is required; gpuIndex must be >= 0");
        }
        if (!new File(image).exists()) {
            throw new IllegalArgumentException("Image file not found: " + image);
        }

        long gpuBeforeLoad = requireGpuMemoryMiB(gpuIndex);
        System.out.printf(Locale.ROOT,
                "model=%s image=%s gpuIndex=%d warmups=%d iterations=%d pid=%d gpuBeforeLoadMiB=%d%n",
                MODEL_NAME, image, gpuIndex, warmups, iterations, pid, gpuBeforeLoad);
        System.out.println("stage timings are nanoseconds");

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(MODEL_ROOT_DIR, MODEL_NAME, env, gpuIndex);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            PaddleDetectionPredictor predictor = detModel.getPredictor();

            for (int i = 0; i < warmups; i++) {
                RunResult legacy = runOnce(predictor, image, matManager, ndManager, true);
                RunResult optimized = runOnce(predictor, image, matManager, ndManager, false);
                assertSameResults(legacy.results(), optimized.results(), "warmup " + (i + 1));
                assertNoTrackedNativeLeak(matManager, "warmup " + (i + 1));
            }

            System.gc();
            Thread.sleep(200);
            long rssBaseline = getRss(pid);
            long gpuBaseline = requireGpuMemoryMiB(gpuIndex);

            Totals legacyTotals = new Totals();
            Totals optimizedTotals = new Totals();
            List<TextDetectionResult> lastResults = List.of();

            System.out.printf(Locale.ROOT,
                    "baseline rssBytes=%d gpuMiB=%d%n", rssBaseline, gpuBaseline);

            for (int i = 0; i < iterations; i++) {
                RunResult legacy = runOnce(predictor, image, matManager, ndManager, true);
                RunResult optimized = runOnce(predictor, image, matManager, ndManager, false);

                assertSameResults(legacy.results(), optimized.results(), "iteration " + (i + 1));
                assertNoTrackedNativeLeak(matManager, "iteration " + (i + 1));

                legacyTotals.add(legacy);
                optimizedTotals.add(optimized);
                lastResults = optimized.results();

                if (i == 0 || (i + 1) % 10 == 0) {
                    System.out.printf(Locale.ROOT,
                            "iter=%d legacyE2ENanos=%d optimizedE2ENanos=%d boxes=%d trackedMat=%d trackedCloseable=%d%n",
                            i + 1,
                            legacy.e2eNanos(),
                            optimized.e2eNanos(),
                            optimized.results().getFirst().polys().length,
                            matManager.trackedMatCount(),
                            matManager.trackedCloseableCount());
                }
            }

            System.gc();
            Thread.sleep(200);
            long rssFinal = getRss(pid);
            long gpuFinal = requireGpuMemoryMiB(gpuIndex);

            printAverages("legacy", legacyTotals, iterations);
            printAverages("optimized", optimizedTotals, iterations);

            long legacyAvg = legacyTotals.e2eNanos / iterations;
            long optimizedAvg = optimizedTotals.e2eNanos / iterations;
            double speedup = legacyAvg == 0 ? 0.0 : (double) legacyAvg / (double) optimizedAvg;
            TextDetectionResult result = lastResults.getFirst();
            System.out.printf(Locale.ROOT,
                    "correctness boxes=%d firstPoly=%s scores=%s compareWithLegacy=PASS%n",
                    result.polys().length,
                    result.polys().length == 0 ? "[]" : Arrays.deepToString(result.polys()[0]),
                    result.scores());
            System.out.printf(Locale.ROOT,
                    "performance legacyAvgE2ENanos=%d optimizedAvgE2ENanos=%d speedup=%.6f%n",
                    legacyAvg,
                    optimizedAvg,
                    speedup);
            System.out.printf(Locale.ROOT,
                    "memory baselineRssBytes=%d finalRssBytes=%d finalRssDeltaBytes=%d "
                            + "baselineGpuMiB=%d finalGpuMiB=%d finalGpuDeltaMiB=%d leakCheck=PASS%n",
                    rssBaseline,
                    rssFinal,
                    rssFinal - rssBaseline,
                    gpuBaseline,
                    gpuFinal,
                    gpuFinal - gpuBaseline);
        }
    }

    private static RunResult runOnce(PaddleDetectionPredictor predictor,
                                     String image,
                                     MatManager matManager,
                                     NDManager ndManager,
                                     boolean legacy) throws Exception {
        long readStart = System.nanoTime();
        Mat rgbMat = ImageUtil.readToRgb(matManager, image);
        long readNanos = System.nanoTime() - readStart;

        PaddleDetectionPredictor.TimedResult timed = legacy
                ? predictor.batchPredictWithLegacyTimings(List.of(rgbMat), matManager, ndManager,
                null, null, null, null, null, null)
                : predictor.batchPredictWithTimings(List.of(rgbMat), matManager, ndManager,
                null, null, null, null, null, null);

        long e2eNanos = readNanos + timed.timings().totalNanos();
        return new RunResult(timed.results(), timed.timings(), readNanos, e2eNanos);
    }

    private static void assertSameResults(List<TextDetectionResult> expected,
                                          List<TextDetectionResult> actual,
                                          String label) {
        if (expected.size() != actual.size()) {
            throw new IllegalStateException(label + " result size mismatch: expected="
                    + expected.size() + " actual=" + actual.size());
        }
        for (int i = 0; i < expected.size(); i++) {
            TextDetectionResult exp = expected.get(i);
            TextDetectionResult act = actual.get(i);
            if (!Arrays.deepEquals(exp.polys(), act.polys())) {
                throw new IllegalStateException(label + " polys mismatch: expected="
                        + Arrays.deepToString(exp.polys()) + " actual=" + Arrays.deepToString(act.polys()));
            }
            if (exp.scores().size() != act.scores().size()) {
                throw new IllegalStateException(label + " score size mismatch: expected="
                        + exp.scores().size() + " actual=" + act.scores().size());
            }
            for (int j = 0; j < exp.scores().size(); j++) {
                float diff = Math.abs(exp.scores().get(j) - act.scores().get(j));
                if (diff > 1e-6f) {
                    throw new IllegalStateException(label + " score mismatch at " + j
                            + ": expected=" + exp.scores().get(j) + " actual=" + act.scores().get(j));
                }
            }
        }
    }

    private static void assertNoTrackedNativeLeak(MatManager matManager, String label) {
        int trackedMat = matManager.trackedMatCount();
        int trackedCloseable = matManager.trackedCloseableCount();
        if (trackedMat != 0 || trackedCloseable != 0) {
            throw new IllegalStateException(
                    label + " leaked tracked resources: trackedMat=" + trackedMat
                            + " trackedCloseable=" + trackedCloseable);
        }
    }

    private static void printAverages(String label, Totals totals, int iterations) {
        System.out.printf(Locale.ROOT,
                "avg %s readNanos=%d padNanos=%d resizeNanos=%d preprocessNanos=%d "
                        + "tensorCreateNanos=%d gpuInferenceNanos=%d outputReadNanos=%d "
                        + "postprocessNanos=%d cleanupNanos=%d detectorTotalNanos=%d e2eNanos=%d%n",
                label,
                totals.readNanos / iterations,
                totals.padNanos / iterations,
                totals.resizeNanos / iterations,
                totals.preprocessNanos / iterations,
                totals.tensorCreateNanos / iterations,
                totals.inferenceNanos / iterations,
                totals.outputReadNanos / iterations,
                totals.postprocessNanos / iterations,
                totals.cleanupNanos / iterations,
                totals.detectorTotalNanos / iterations,
                totals.e2eNanos / iterations);
    }

    private static long getRss(long pid) {
        try {
            Process p = new ProcessBuilder("tasklist", "/NH", "/FI", "PID eq " + pid).start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = br.readLine();
                while (line != null) {
                    if (line.contains(String.valueOf(pid))) {
                        Matcher m = RSS_PATTERN.matcher(line);
                        if (m.find()) {
                            return Long.parseLong(m.group(1).replace(",", "")) * 1024L;
                        }
                    }
                    line = br.readLine();
                }
            }
            p.waitFor();
        } catch (Exception ignored) {
            return -1L;
        }
        return -1L;
    }

    private static long requireGpuMemoryMiB(int gpuIndex) throws Exception {
        Process p = new ProcessBuilder(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                String.valueOf(gpuIndex)).start();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line = br.readLine();
            int exit = p.waitFor();
            if (exit != 0 || line == null || line.isBlank()) {
                throw new IllegalStateException("Unable to read GPU memory with nvidia-smi");
            }
            return Long.parseLong(line.trim());
        }
    }

    private record RunResult(List<TextDetectionResult> results,
                             PaddleDetectionPredictor.StageTimings timings,
                             long readNanos,
                             long e2eNanos) {
    }

    private static final class Totals {
        long readNanos;
        long padNanos;
        long resizeNanos;
        long preprocessNanos;
        long tensorCreateNanos;
        long inferenceNanos;
        long outputReadNanos;
        long postprocessNanos;
        long cleanupNanos;
        long detectorTotalNanos;
        long e2eNanos;

        void add(RunResult result) {
            PaddleDetectionPredictor.StageTimings t = result.timings();
            readNanos += result.readNanos();
            padNanos += t.padNanos();
            resizeNanos += t.resizeNanos();
            preprocessNanos += t.preprocessNanos();
            tensorCreateNanos += t.tensorCreateNanos();
            inferenceNanos += t.inferenceNanos();
            outputReadNanos += t.outputReadNanos();
            postprocessNanos += t.postprocessNanos();
            cleanupNanos += t.cleanupNanos();
            detectorTotalNanos += t.totalNanos();
            e2eNanos += result.e2eNanos();
        }
    }
}
