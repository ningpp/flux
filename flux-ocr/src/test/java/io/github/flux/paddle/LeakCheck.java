package io.github.flux.paddle;

import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.pipeline.OCRPipeline;
import io.github.flux.pipeline.OCRPipelineResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.ProcessBuilder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 内存泄露 / 性能验证工具（针对 PP-OCRv6_medium_det / PP-OCRv6_medium_rec）。
 *
 * 用法:
 *   LeakCheck pipeline <gpuIndex> <iterations> <image>
 *       - 全流水线（det+rec）对单图反复推理，按进程工作集(RSS)观察原生内存是否稳定
 *   LeakCheck det <gpuIndex> <iterations> <image>
 *       - 用长期存活的 MatManager 反复调用 PaddleDetectionPredictor，观察 trackedMatCount 是否回归基线
 *
 * 例:
 *   LeakCheck pipeline -1 20 D:\tmp\twgx-gq.png
 *   LeakCheck det      0 20 D:\tmp\twgx-gq.png
 */
public class LeakCheck {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final Pattern RSS_PATTERN = Pattern.compile("([\\d,]+)\\s*K");

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String mode = args.length > 0 ? args[0] : "pipeline";
        int gpuIndex = args.length > 1 ? Integer.parseInt(args[1]) : -1;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        String image = args.length > 3 ? args[3] : "D:\\tmp\\twgx-gq.png";
        long pid = ProcessHandle.current().pid();

        System.out.printf("=== LeakCheck mode=%s gpuIndex=%d iterations=%d image=%s pid=%d ===%n",
                mode, gpuIndex, iterations, image, pid);

        if ("det".equals(mode)) {
            runDet(gpuIndex, iterations, image, pid);
        } else {
            runPipeline(gpuIndex, iterations, image, pid);
        }
    }

    private static void runPipeline(int gpuIndex, int iterations, String image, long pid) throws Exception {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex)) {

            OCRPipeline pipeline = new OCRPipeline(detModel, recModel, null, null);
            Map<String, Object> params = new HashMap<>();
            params.put("recognitionBatchSize", 4);

            // warmup
            pipeline.predict(List.of(image), params);
            System.gc();

            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                List<List<OCRPipelineResult>> results = pipeline.predict(List.of(image), params);
                long t1 = System.nanoTime();

                int textCount = 0;
                StringBuilder sample = new StringBuilder();
                for (List<OCRPipelineResult> page : results) {
                    for (OCRPipelineResult r : page) {
                        // 顶层文本行
                        if (r.recResults() != null) {
                            for (var rr : r.recResults()) {
                                textCount++;
                                if (sample.length() < 100) {
                                    sample.append(rr.text()).append(" | ");
                                }
                            }
                        }
                        // 布局区域内的文本行
                        if (r.layoutRegions() != null) {
                            for (var region : r.layoutRegions()) {
                                if (region.textResults() != null) {
                                    for (var tr : region.textResults()) {
                                        if (tr.recResults() != null) {
                                            for (var rr : tr.recResults()) {
                                                textCount++;
                                                if (sample.length() < 100) {
                                                    sample.append(rr.text()).append(" | ");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                long rss = getRss(pid);
                System.gc();
                long rssGc = getRss(pid);
                System.out.printf("iter=%2d  time=%7.1f ms  textLines=%4d  rss=%6.1f MB  rssAfterGC=%6.1f MB  sample=%s%n",
                        i + 1, (t1 - t0) / 1e6, textCount,
                        rss / 1024.0 / 1024.0, rssGc / 1024.0 / 1024.0, sample);
            }
        }
    }

    private static void runDet(int gpuIndex, int iterations, String image, long pid) throws Exception {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex)) {

            MatManager matManager = new MatManager(); // 长期存活
            Mat srcMat = ImageUtil.readToRgb(matManager, image);

            Map<String, Object> params = new HashMap<>();
            for (int i = 0; i < iterations; i++) {
                // 每轮克隆一张 srcMat 交给 predictor 消费，srcMat 本身长期存活，
                // 以便观察 matManager 是否随迭代累积（回归基线应为 1）。
                Mat srcClone = matManager.cloneMat(srcMat);
                try (ai.djl.ndarray.NDManager ndManager = ai.djl.ndarray.NDManager.newBaseManager()) {
                    List<PreProcessResult> pre = List.of(new PreProcessResult(srcClone, null));
                    long t0 = System.nanoTime();
                    List<TextDetectionResult> dets = detModel.doBatchPredict(pre, matManager, ndManager, params);
                    long t1 = System.nanoTime();

                    int tracked = matManager.trackedMatCount();
                    long rss = getRss(pid);
                    System.out.printf("iter=%2d  time=%7.1f ms  boxes=%4d  trackedMat=%3d  rss=%6.1f MB%n",
                            i + 1, (t1 - t0) / 1e6, dets.size(), tracked, rss / 1024.0 / 1024.0);
                }
                matManager.release(srcClone);
            }

            matManager.release(srcMat);
            matManager.close();
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
        } catch (Exception e) {
            System.err.println("getRss failed: " + e.getMessage());
        }
        return 0;
    }
}
