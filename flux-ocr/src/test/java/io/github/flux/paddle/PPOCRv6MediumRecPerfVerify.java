package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.RecognitionResult;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.paddle.predictor.PaddleRecognitionPredictor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PPOCRv6MediumRecPerfVerify {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String MODEL_NAME = "PP-OCRv6_medium_rec";
    private static final String DEFAULT_IMAGE = "D:\\tmp\\text-rec-20260708194657.png";
    private static final String EXPECTED_TEXT = "静夜思";
    private static final Pattern RSS_PATTERN = Pattern.compile("([\\d,]+)\\s*K");

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String image = args.length > 0 ? args[0] : DEFAULT_IMAGE;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 100000;
        int warmups = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        int gpuIndex = args.length > 3 ? Integer.parseInt(args[3]) : 0;
        long pid = ProcessHandle.current().pid();

        if (!new File(image).exists()) {
            throw new IllegalArgumentException("Image file not found: " + image);
        }
        long gpuBeforeLoad = requireGpuMemoryMiB(gpuIndex);

        System.out.printf(Locale.ROOT,
                "model=%s image=%s gpuIndex=%d warmups=%d iterations=%d pid=%d gpuBeforeLoadMiB=%d%n",
                MODEL_NAME, image, gpuIndex, warmups, iterations, pid, gpuBeforeLoad);

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextRecognitionModel recModel = new TextRecognitionModel(MODEL_ROOT_DIR, MODEL_NAME, env, gpuIndex);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            PaddleRecognitionPredictor predictor = recModel.getPredictor();

            for (int i = 0; i < warmups; i++) {
                Mat rgbMat = ImageUtil.readToRgb(matManager, image);
                predictor.batchPredictWithTimings(List.of(rgbMat), matManager, ndManager);
                assertNoTrackedNativeLeak(matManager, "warmup " + (i + 1));
            }

            System.gc();
            Thread.sleep(200);
            long rssBaseline = getRss(pid);
            long gpuBaseline = requireGpuMemoryMiB(gpuIndex);

            long readSum = 0L;
            long padSum = 0L;
            long preprocessSum = 0L;
            long tensorSum = 0L;
            long inferenceSum = 0L;
            long outputReadSum = 0L;
            long postprocessSum = 0L;
            long cleanupSum = 0L;
            long recTotalSum = 0L;
            long e2eSum = 0L;

            System.out.printf(Locale.ROOT,
                    "baseline rssBytes=%d gpuMiB=%d%n", rssBaseline, gpuBaseline);
            System.out.println("stage timings are nanoseconds");

            for (int i = 0; i < iterations; i++) {
                long readStart = System.nanoTime();
                Mat rgbMat = ImageUtil.readToRgb(matManager, image);
                long readNanos = System.nanoTime() - readStart;

                PaddleRecognitionPredictor.TimedResult timed =
                    predictor.batchPredictWithTimings(List.of(rgbMat), matManager, ndManager);
                PaddleRecognitionPredictor.StageTimings t = timed.timings();
                long e2eNanos = readNanos + t.totalNanos();

                RecognitionResult result = timed.results().getFirst().getFirst();
                if (!EXPECTED_TEXT.equals(result.text())) {
                    throw new IllegalStateException(
                        "Unexpected recognition text: expected=" + EXPECTED_TEXT + " actual=" + result.text());
                }
                assertNoTrackedNativeLeak(matManager, "iteration " + (i + 1));

                readSum += readNanos;
                padSum += t.padNanos();
                preprocessSum += t.preprocessNanos();
                tensorSum += t.tensorCreateNanos();
                inferenceSum += t.inferenceNanos();
                outputReadSum += t.outputReadNanos();
                postprocessSum += t.postprocessNanos();
                cleanupSum += t.cleanupNanos();
                recTotalSum += t.totalNanos();
                e2eSum += e2eNanos;

                if (i % 200 == 0) {
                    System.out.printf(Locale.ROOT,
                        "iter=%02d readNanos=%d padNanos=%d preprocessNanos=%d tensorCreateNanos=%d "
                            + "gpuInferenceNanos=%d outputReadNanos=%d postprocessNanos=%d cleanupNanos=%d "
                            + "recTotalNanos=%d e2eNanos=%d text=%s score=%.9f trackedMat=%d trackedCloseable=%d%n",
                        i + 1,
                        readNanos,
                        t.padNanos(),
                        t.preprocessNanos(),
                        t.tensorCreateNanos(),
                        t.inferenceNanos(),
                        t.outputReadNanos(),
                        t.postprocessNanos(),
                        t.cleanupNanos(),
                        t.totalNanos(),
                        e2eNanos,
                        result.text(),
                        result.scores()[0],
                        matManager.trackedMatCount(),
                        matManager.trackedCloseableCount());
                }
            }

            System.gc();
            Thread.sleep(200);
            long rssFinal = getRss(pid);
            long gpuFinal = requireGpuMemoryMiB(gpuIndex);
            System.out.printf(Locale.ROOT,
                    "avg readNanos=%d padNanos=%d preprocessNanos=%d tensorCreateNanos=%d "
                            + "gpuInferenceNanos=%d outputReadNanos=%d postprocessNanos=%d cleanupNanos=%d "
                            + "recTotalNanos=%d e2eNanos=%d%n",
                    readSum / iterations,
                    padSum / iterations,
                    preprocessSum / iterations,
                    tensorSum / iterations,
                    inferenceSum / iterations,
                    outputReadSum / iterations,
                    postprocessSum / iterations,
                    cleanupSum / iterations,
                    recTotalSum / iterations,
                    e2eSum / iterations);
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

    private static void assertNoTrackedNativeLeak(MatManager matManager, String label) {
        int trackedMat = matManager.trackedMatCount();
        int trackedCloseable = matManager.trackedCloseableCount();
        if (trackedMat != 0 || trackedCloseable != 0) {
            throw new IllegalStateException(
                    label + " leaked tracked resources: trackedMat=" + trackedMat
                            + " trackedCloseable=" + trackedCloseable);
        }
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
        if (gpuIndex < 0) {
            throw new IllegalArgumentException("GPU inference is required; gpuIndex must be >= 0");
        }
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
}
