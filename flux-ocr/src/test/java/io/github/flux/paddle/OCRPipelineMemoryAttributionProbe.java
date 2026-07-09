package io.github.flux.paddle;

import ai.onnxruntime.OrtEnvironment;
import io.github.flux.bytedeco.OpenCVImage;
import io.github.flux.core.MatManager;
import io.github.flux.model.DocOrientationClassifyModel;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.LayoutModel;
import io.github.flux.model.TableModel;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.pipeline.LayoutRegionResult;
import io.github.flux.pipeline.OCRPipeline;
import io.github.flux.pipeline.OCRPipelineResult;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.Mat;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Real-process memory attribution probe for OCRPipeline.
 *
 * <p>This probe intentionally keeps one shared pipeline and does not close models
 * between requests. It separates resident model-load deltas from per-request
 * temporary growth and post-request release behavior.
 */
public class OCRPipelineMemoryAttributionProbe {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final int DPI = 300;
    private static final int GPU_INDEX = 0;
    private static final int TABLE_DECODER_GPU_INDEX = -1;
    private static final int TABLE_MAX_TOKENS = 768;

    private static final String[] PDF_FILES = {
            "E:\\flux-data\\2606.13108_zh_CN.pdf",
            "E:\\flux-data\\2606.13108.pdf",
            "E:\\flux-data\\2606.13392_zh_CN.pdf",
            "E:\\flux-data\\2606.13392.pdf"
    };

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        if (args.length == 0) {
            usage();
            return;
        }

        switch (args[0]) {
            case "load" -> measureModelLoad();
            case "request" -> measureRepeatedRequest(args);
            default -> usage();
        }
    }

    static List<Integer> parsePageRange(String value) {
        String[] parts = value.split("-");
        int start = Integer.parseInt(parts[0].trim());
        int end = parts.length == 1 ? start : Integer.parseInt(parts[1].trim());
        if (start < 1 || end < start) {
            throw new IllegalArgumentException("Invalid one-based page range: " + value);
        }
        List<Integer> pages = new ArrayList<>();
        for (int page = start; page <= end; page++) {
            pages.add(page);
        }
        return pages;
    }

    static double percent(double value, double total) {
        if (Math.abs(total) < 1e-9d) {
            return 0d;
        }
        return value * 100d / total;
    }

    record RequestSettings(String pdfPath,
                           String pagesSpec,
                           int repeats,
                           int pageBatchSize,
                           int recognitionBatchSize,
                           int formulaBatchSize,
                           int detectionBatchSize,
                           int layoutBatchSize) {
    }

    static RequestSettings parseRequestSettings(String[] args) {
        String pdfPath = args.length > 1 ? args[1] : PDF_FILES[0];
        String pagesSpec = args.length > 2 ? args[2] : "1-4";
        int repeats = args.length > 3 ? Integer.parseInt(args[3]) : 2;
        int pageBatchSize = args.length > 4 ? Integer.parseInt(args[4]) : 4;
        int recognitionBatchSize = args.length > 5 ? Integer.parseInt(args[5]) : 8;
        int formulaBatchSize = args.length > 6 ? Integer.parseInt(args[6]) : 4;
        int detectionBatchSize = args.length > 7 ? Integer.parseInt(args[7]) : 1;
        int layoutBatchSize = args.length > 8 ? Integer.parseInt(args[8]) : 1;
        return new RequestSettings(pdfPath, pagesSpec, repeats, pageBatchSize,
                recognitionBatchSize, formulaBatchSize, detectionBatchSize, layoutBatchSize);
    }

    private static void measureModelLoad() throws Exception {
        List<MemoryPoint> points = new ArrayList<>();
        List<AutoCloseable> closeables = new ArrayList<>();

        record(points, "baseline");
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            closeables.add(env);
            record(points, "after-env");

            TextDetectionModel detModel = new TextDetectionModel(
                    OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, GPU_INDEX);
            closeables.add(detModel);
            record(points, "after-det");

            TextRecognitionModel recModel = new TextRecognitionModel(
                    OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, GPU_INDEX);
            closeables.add(recModel);
            record(points, "after-rec");

            DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(
                    OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, GPU_INDEX);
            closeables.add(docOriModel);
            record(points, "after-doc-ori");

            TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(
                    OCR_MODEL_DIR, "PP-LCNet_x1_0_textline_ori", env, GPU_INDEX);
            closeables.add(textLineOriModel);
            record(points, "after-textline-ori");

            LayoutModel layoutModel = new LayoutModel(
                    LAYOUT_MODEL_DIR, "PP-DocLayoutV3", GPU_INDEX, env);
            closeables.add(layoutModel);
            record(points, "after-layout");

            FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(
                    FORMULA_MODEL_DIR, "pix2text-mfr-1.5", GPU_INDEX, env);
            closeables.add(formulaModel);
            record(points, "after-formula");

            TableModel tableModel = new TableModel(
                    FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env,
                    Map.of(
                            "unirec.decoderGpuIndex", TABLE_DECODER_GPU_INDEX,
                            "unirec.maxTokens", TABLE_MAX_TOKENS
                    ));
            closeables.add(tableModel);
            record(points, "after-table");

            printLoadAttribution(points);
        } finally {
            closeReverse(closeables);
            requestGc();
            record(points, "after-close-and-gc");
        }
    }

    private static void measureRepeatedRequest(String[] args) throws Exception {
        RequestSettings settings = parseRequestSettings(args);

        List<MemoryPoint> points = new ArrayList<>();
        List<AutoCloseable> closeables = new ArrayList<>();
        List<Integer> pages = parsePageRange(settings.pagesSpec());
        List<String> imagePaths = new ArrayList<>();

        record(points, "request:baseline");
        try {
            imagePaths = convertPdfPagesToImages(settings.pdfPath(), pages);
            record(points, "request:after-pdf-render");

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            closeables.add(env);
            TextDetectionModel detModel = new TextDetectionModel(
                    OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, GPU_INDEX);
            TextRecognitionModel recModel = new TextRecognitionModel(
                    OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, GPU_INDEX);
            DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(
                    OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, GPU_INDEX);
            TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(
                    OCR_MODEL_DIR, "PP-LCNet_x1_0_textline_ori", env, GPU_INDEX);
            LayoutModel layoutModel = new LayoutModel(
                    LAYOUT_MODEL_DIR, "PP-DocLayoutV3", GPU_INDEX, env);
            FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(
                    FORMULA_MODEL_DIR, "pix2text-mfr-1.5", GPU_INDEX, env);
            TableModel tableModel = new TableModel(
                    FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env,
                    Map.of(
                            "unirec.decoderGpuIndex", TABLE_DECODER_GPU_INDEX,
                            "unirec.maxTokens", TABLE_MAX_TOKENS
                    ));
            closeables.add(detModel);
            closeables.add(recModel);
            closeables.add(docOriModel);
            closeables.add(textLineOriModel);
            closeables.add(layoutModel);
            closeables.add(formulaModel);
            closeables.add(tableModel);
            record(points, "request:after-model-load");

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel, docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);

            Map<String, Object> params = new HashMap<>();
            params.put("layoutBatchSize", settings.layoutBatchSize());
            params.put("detectionBatchSize", settings.detectionBatchSize());
            params.put("recognitionBatchSize", settings.recognitionBatchSize());
            params.put("formulaBatchSize", settings.formulaBatchSize());
            params.put("tableBatchSize", 1);
            params.put("memoryObserver", (Consumer<String>) stage ->
                    record(points, "request-stage:" + stage));

            for (int repeat = 1; repeat <= settings.repeats(); repeat++) {
                int successCount = 0;
                long ocrNanos = 0L;
                record(points, "request-" + repeat + ":before");
                for (int start = 0; start < imagePaths.size(); start += settings.pageBatchSize()) {
                    int end = Math.min(start + settings.pageBatchSize(), imagePaths.size());
                    List<String> batch = imagePaths.subList(start, end);
                    LocalDateTime batchStart = LocalDateTime.now();
                    List<List<OCRPipelineResult>> results = pipeline.predict(batch, params);
                    ocrNanos += Duration.between(batchStart, LocalDateTime.now()).toNanos();
                    successCount += results.size();
                    printResultSummary(repeat, start, end, results);
                    record(points, "request-" + repeat + ":after-batch-" + (start + 1) + "-" + end);
                }
                record(points, "request-" + repeat + ":after");
                requestGc();
                record(points, "request-" + repeat + ":after-gc");
                System.out.printf(Locale.ROOT,
                        "REQUEST %d done pages=%d avgOcrMs=%.2f%n",
                        repeat, successCount, ocrNanos / 1_000_000d / Math.max(1, successCount));
            }

            printRequestAttribution(points);
        } finally {
            for (String imagePath : imagePaths) {
                try {
                    Files.deleteIfExists(Path.of(imagePath));
                } catch (Exception ignored) {
                }
            }
            closeReverse(closeables);
            requestGc();
            record(points, "request:after-close-and-gc");
        }
    }

    private static List<String> convertPdfPagesToImages(String pdfFile, List<Integer> pages) throws Exception {
        List<String> imagePaths = new ArrayList<>();
        try (PDDocument document = Loader.loadPDF(new File(pdfFile));
             MatManager matManager = new MatManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            String baseName = new File(pdfFile).getName().replace(".pdf", "");

            for (int pageNumber : pages) {
                BufferedImage bufferedImage = renderer.renderImageWithDPI(pageNumber - 1, DPI, ImageType.RGB);
                Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
                Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);
                String imagePath = System.getProperty("java.io.tmpdir") + File.separator
                        + baseName + "_probe_page_" + pageNumber + ".png";
                org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, rgbMat);
                imagePaths.add(imagePath);
                matManager.release(mat);
                matManager.release(rgbMat);
            }
        }
        return imagePaths;
    }

    private static void printResultSummary(int repeat,
                                           int start,
                                           int end,
                                           List<List<OCRPipelineResult>> results) {
        int regionCount = 0;
        int textLineCount = 0;
        int formulaCount = 0;
        int tableCount = 0;
        for (List<OCRPipelineResult> pageResults : results) {
            for (OCRPipelineResult result : pageResults) {
                if (result.layoutRegions() == null) {
                    textLineCount += result.recResults() == null ? 0 : result.recResults().size();
                    continue;
                }
                for (LayoutRegionResult region : result.layoutRegions()) {
                    regionCount++;
                    switch (region.regionType()) {
                        case "text" -> textLineCount += region.textResults() == null ? 0 : region.textResults().size();
                        case "formula" -> formulaCount++;
                        case "table" -> tableCount++;
                        default -> {
                        }
                    }
                }
            }
        }
        System.out.printf("REQUEST %d batch %d-%d regions=%d textLines=%d formulas=%d tables=%d%n",
                repeat, start + 1, end, regionCount, textLineCount, formulaCount, tableCount);
    }

    private static void printLoadAttribution(List<MemoryPoint> points) {
        Map<String, String> stageNames = new LinkedHashMap<>();
        stageNames.put("after-env", "ort-env");
        stageNames.put("after-det", "text-det");
        stageNames.put("after-rec", "text-rec");
        stageNames.put("after-doc-ori", "doc-ori");
        stageNames.put("after-textline-ori", "textline-ori");
        stageNames.put("after-layout", "layout");
        stageNames.put("after-formula", "formula");
        stageNames.put("after-table", "table");

        double positivePrivateTotal = 0d;
        List<MemoryDelta> deltas = new ArrayList<>();
        for (int i = 1; i < points.size(); i++) {
            MemoryPoint previous = points.get(i - 1);
            MemoryPoint current = points.get(i);
            String name = stageNames.getOrDefault(current.label(), current.label());
            MemoryDelta delta = new MemoryDelta(name, current.privateGb() - previous.privateGb(),
                    current.workingSetGb() - previous.workingSetGb());
            deltas.add(delta);
            if (delta.privateGb() > 0d) {
                positivePrivateTotal += delta.privateGb();
            }
        }

        System.out.println();
        System.out.println("========== MODEL LOAD ATTRIBUTION ==========");
        System.out.printf(Locale.ROOT, "%-16s %12s %12s %9s%n",
                "component", "privateGB", "workingGB", "private%");
        for (MemoryDelta delta : deltas) {
            System.out.printf(Locale.ROOT, "%-16s %12.2f %12.2f %8.1f%%%n",
                    delta.label(), delta.privateGb(), delta.workingSetGb(),
                    percent(Math.max(0d, delta.privateGb()), positivePrivateTotal));
        }
        MemoryPoint first = points.getFirst();
        MemoryPoint last = points.getLast();
        System.out.printf(Locale.ROOT,
                "TOTAL resident delta: private %.2f GB, working %.2f GB%n",
                last.privateGb() - first.privateGb(),
                last.workingSetGb() - first.workingSetGb());
    }

    private static void printRequestAttribution(List<MemoryPoint> points) {
        System.out.println();
        System.out.println("========== REQUEST MEMORY TIMELINE ==========");
        System.out.printf(Locale.ROOT, "%-34s %10s %10s %10s%n",
                "stage", "heapGB", "workGB", "privGB");
        for (MemoryPoint point : points) {
            System.out.printf(Locale.ROOT, "%-34s %10.2f %10.2f %10.2f%n",
                    point.label(), point.heapGb(), point.workingSetGb(), point.privateGb());
        }

        MemoryPoint peakPrivate = points.stream()
                .max((a, b) -> Double.compare(a.privateGb(), b.privateGb()))
                .orElse(points.getLast());
        MemoryPoint peakWorking = points.stream()
                .max((a, b) -> Double.compare(a.workingSetGb(), b.workingSetGb()))
                .orElse(points.getLast());
        System.out.printf(Locale.ROOT,
                "PEAK private: %.2f GB at %s; PEAK working: %.2f GB at %s%n",
                peakPrivate.privateGb(), peakPrivate.label(),
                peakWorking.workingSetGb(), peakWorking.label());
    }

    private static void record(List<MemoryPoint> points, String label) {
        MemoryPoint point = MemoryPoint.capture(label);
        points.add(point);
        System.out.printf(Locale.ROOT,
                "MEMORY %-34s heapGB=%.2f workingGB=%.2f privateGB=%.2f%n",
                point.label(), point.heapGb(), point.workingSetGb(), point.privateGb());
    }

    private static void requestGc() {
        System.gc();
        System.runFinalization();
        try {
            Thread.sleep(1500L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void closeReverse(List<AutoCloseable> closeables) {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            IOUtil.close(closeables.get(i));
        }
    }

    private static ProcessMemorySnapshot windowsProcessMemory() {
        long pid = ProcessHandle.current().pid();
        String command = "$p=Get-Process -Id " + pid
                + "; [string]::Format([Globalization.CultureInfo]::InvariantCulture,"
                + " '{0:F6} {1:F6}', $p.WorkingSet64/1GB, $p.PrivateMemorySize64/1GB)";
        try {
            Process process = new ProcessBuilder(
                    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)
                    .redirectErrorStream(true)
                    .start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            int exit = process.waitFor();
            if (exit == 0 && !output.isBlank()) {
                String[] parts = output.split("\\s+");
                if (parts.length >= 2) {
                    return new ProcessMemorySnapshot(
                            Double.parseDouble(parts[0]),
                            Double.parseDouble(parts[1]));
                }
            }
        } catch (Exception ignored) {
            // Keep probe running even if process memory sampling is unavailable.
        }
        return new ProcessMemorySnapshot(0d, 0d);
    }

    private static void usage() {
        System.out.println("""
                Usage:
                  load
                  request [pdfPath] [pages] [repeats] [pageBatchSize] [recognitionBatchSize] [formulaBatchSize] [detectionBatchSize] [layoutBatchSize]

                Examples:
                  load
                  request E:\\flux-data\\2606.13108_zh_CN.pdf 1-4 2 4 8 4
                  request E:\\flux-data\\2606.13108_zh_CN.pdf 1-4 2 4 8 4 1 1
                """);
    }

    private record MemoryPoint(String label, double heapGb, double workingSetGb, double privateGb) {
        static MemoryPoint capture(String label) {
            Runtime runtime = Runtime.getRuntime();
            double heapUsedGb = (runtime.totalMemory() - runtime.freeMemory()) / 1024d / 1024d / 1024d;
            ProcessMemorySnapshot process = windowsProcessMemory();
            return new MemoryPoint(label, heapUsedGb, process.workingSetGb(), process.privateGb());
        }
    }

    private record MemoryDelta(String label, double privateGb, double workingSetGb) {
    }

    private record ProcessMemorySnapshot(double workingSetGb, double privateGb) {
    }
}
