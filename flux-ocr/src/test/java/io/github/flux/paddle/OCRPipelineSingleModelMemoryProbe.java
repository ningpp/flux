package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.bytedeco.OpenCVImage;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.LayoutModel;
import io.github.flux.model.TableModel;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Diagnostic entry point for isolating OCRPipeline memory spikes.
 *
 * <p>Run extract first to save the exact page and layout-region images, then run
 * formula/table/text-det/text-rec as separate JVM commands so each model's native
 * memory profile can be observed independently.
 */
public class OCRPipelineSingleModelMemoryProbe {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final int DPI = 300;
    private static final int GPU_INDEX = 0;
    private static final int TABLE_DECODER_GPU_INDEX = -1;
    private static final int TABLE_MAX_TOKENS = 768;

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        if (args.length < 1) {
            usage();
            return;
        }

        String mode = args[0];
        switch (mode) {
            case "extract" -> {
                if (args.length != 4) {
                    usage();
                    return;
                }
                extractLayoutRegions(Paths.get(args[1]), parsePageRange(args[2]), Paths.get(args[3]));
            }
            case "formula" -> runFormula(Paths.get(args[1]), parseBatchSize(args, 2, 4));
            case "table" -> runTable(Paths.get(args[1]), parseBatchSize(args, 2, 1));
            case "text-det" -> runTextDetection(Paths.get(args[1]), parseBatchSize(args, 2, 8));
            case "text-rec" -> runTextRecognition(Paths.get(args[1]), parseBatchSize(args, 2, 8));
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

    static String classifyLabel(String label) {
        if (label == null) {
            return "text";
        }
        String normalized = label.toLowerCase(Locale.ROOT);
        if (normalized.contains("formula") || normalized.contains("equation")) {
            return "formula";
        }
        if (normalized.contains("table")) {
            return "table";
        }
        if (normalized.contains("image") || normalized.contains("figure") || normalized.contains("chart")) {
            return "image";
        }
        return "text";
    }

    private static void extractLayoutRegions(Path pdfPath, List<Integer> pages, Path outDir) throws Exception {
        Path pageDir = outDir.resolve("pages");
        Path formulaDir = outDir.resolve("formula");
        Path tableDir = outDir.resolve("table");
        Path textDir = outDir.resolve("text");
        Path imageDir = outDir.resolve("image");
        Files.createDirectories(pageDir);
        Files.createDirectories(formulaDir);
        Files.createDirectories(tableDir);
        Files.createDirectories(textDir);
        Files.createDirectories(imageDir);

        List<String> manifest = new ArrayList<>();
        manifest.add("type,page,region,label,score,width,height,path");

        logMemory("extract:start");
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, "PP-DocLayoutV3", GPU_INDEX, env);
             PDDocument document = Loader.loadPDF(pdfPath.toFile());
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            for (int pageNumber : pages) {
                BufferedImage bufferedImage = renderer.renderImageWithDPI(pageNumber - 1, DPI, ImageType.RGB);
                Mat bgrPage = OpenCVImage.image2Mat(matManager, bufferedImage);
                Path pagePath = pageDir.resolve("page_%03d.png".formatted(pageNumber));
                Imgcodecs.imwrite(pagePath.toString(), bgrPage);

                Mat rgbPage = matManager.newMat();
                ProcessedMat layoutInput = null;
                try {
                    Imgproc.cvtColor(bgrPage, rgbPage, Imgproc.COLOR_BGR2RGB);
                    layoutInput = layoutModel.processRgb(matManager, rgbPage, ndManager);
                    List<ObjectDetectionResult> regions = layoutModel.batchPredict(
                            List.of(layoutInput), 1, matManager, ndManager, Map.of()).getFirst();
                    for (int regionIndex = 0; regionIndex < regions.size(); regionIndex++) {
                        ObjectDetectionResult region = regions.get(regionIndex);
                        String type = classifyLabel(region.label());
                        Path cropDir = switch (type) {
                            case "formula" -> formulaDir;
                            case "table" -> tableDir;
                            case "image" -> imageDir;
                            default -> textDir;
                        };
                        Mat crop = cropRegion(matManager, bgrPage, region.coordinate());
                        try {
                            Path cropPath = cropDir.resolve("page_%03d_region_%03d_%s.png".formatted(
                                    pageNumber, regionIndex, sanitize(region.label())));
                            Imgcodecs.imwrite(cropPath.toString(), crop);
                            manifest.add("%s,%d,%d,%s,%.6f,%d,%d,%s".formatted(
                                    type,
                                    pageNumber,
                                    regionIndex,
                                    sanitize(region.label()),
                                    region.score(),
                                    crop.cols(),
                                    crop.rows(),
                                    cropPath.toAbsolutePath()));
                        } finally {
                            matManager.release(crop);
                        }
                    }
                    System.out.printf("Extracted page %d: %d regions%n", pageNumber, regions.size());
                    logMemory("extract:page-" + pageNumber);
                } finally {
                    if (layoutInput != null) {
                        layoutInput.release(matManager);
                    }
                    matManager.release(rgbPage);
                    matManager.release(bgrPage);
                }
            }
        }
        Files.write(outDir.resolve("manifest.csv"), manifest, StandardCharsets.UTF_8);
        logMemory("extract:end");
        System.out.println("Saved probe artifacts: " + outDir.toAbsolutePath());
    }

    private static void runFormula(Path outDir, int batchSize) throws Exception {
        List<Path> images = listPngs(outDir.resolve("formula"));
        System.out.printf("Formula crops: %d, batchSize=%d%n", images.size(), batchSize);
        logMemory("formula:start");
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             FormulaRecognitionModel model = new FormulaRecognitionModel(
                     FORMULA_MODEL_DIR, "pix2text-mfr-1.5", GPU_INDEX, env);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            logMemory("formula:after-model-load");
            int batchIndex = 0;
            for (List<Path> batch : split(images, batchSize)) {
                List<PreProcessResult> inputs = new ArrayList<>();
                try {
                    for (Path image : batch) {
                        inputs.add(model.processRgb(matManager, ImageUtil.readToRgb(matManager, image.toString()), ndManager));
                    }
                    logMemory("formula:before-batch-" + batchIndex);
                    List<TextResult> results = model.batchPredict(inputs, batchSize, matManager, ndManager, Map.of());
                    System.out.printf("Formula batch %d done: inputs=%d, results=%d%n",
                            batchIndex, batch.size(), results.size());
                    logMemory("formula:after-batch-" + batchIndex);
                } finally {
                    for (PreProcessResult input : inputs) {
                        releasePreProcessResult(matManager, input);
                    }
                }
                batchIndex++;
            }
        }
        logMemory("formula:end");
    }

    private static void runTable(Path outDir, int batchSize) throws Exception {
        List<Path> images = listPngs(outDir.resolve("table"));
        System.out.printf("Table crops: %d, batchSize=%d%n", images.size(), batchSize);
        logMemory("table:start");
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TableModel model = new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env,
                     Map.of(
                             "unirec.decoderGpuIndex", TABLE_DECODER_GPU_INDEX,
                             "unirec.maxTokens", TABLE_MAX_TOKENS
                     ));
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            logMemory("table:after-model-load");
            int batchIndex = 0;
            for (List<Path> batch : split(images, batchSize)) {
                List<PreProcessResult> inputs = new ArrayList<>();
                try {
                    for (Path image : batch) {
                        inputs.add(model.processRgb(matManager, ImageUtil.readToRgb(matManager, image.toString()), ndManager));
                    }
                    logMemory("table:before-batch-" + batchIndex);
                    List<TableResult> results = model.batchPredict(inputs, batchSize, matManager, ndManager, Map.of());
                    System.out.printf("Table batch %d done: inputs=%d, results=%d%n",
                            batchIndex, batch.size(), results.size());
                    logMemory("table:after-batch-" + batchIndex);
                } finally {
                    for (PreProcessResult input : inputs) {
                        releasePreProcessResult(matManager, input);
                    }
                }
                batchIndex++;
            }
        }
        logMemory("table:end");
    }

    private static void runTextDetection(Path outDir, int batchSize) throws Exception {
        List<Path> images = listPngs(outDir.resolve("text"));
        Path lineDir = outDir.resolve("line");
        Files.createDirectories(lineDir);
        System.out.printf("Text region crops: %d, batchSize=%d%n", images.size(), batchSize);
        logMemory("text-det:start");
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel model = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, GPU_INDEX);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            logMemory("text-det:after-model-load");
            int batchIndex = 0;
            for (List<Path> batch : split(images, batchSize)) {
                List<Mat> sourceImages = new ArrayList<>();
                List<PreProcessResult> inputs = new ArrayList<>();
                try {
                    for (Path image : batch) {
                        Mat rgb = ImageUtil.readToRgb(matManager, image.toString());
                        sourceImages.add(rgb);
                        inputs.add(new PreProcessResult(matManager.cloneMat(rgb), null));
                    }
                    logMemory("text-det:before-batch-" + batchIndex);
                    List<TextDetectionResult> results = model.batchPredict(inputs, batchSize, matManager, ndManager, Map.of());
                    saveLineCrops(batch, sourceImages, results, lineDir, matManager, ndManager);
                    System.out.printf("Text-det batch %d done: inputs=%d, results=%d%n",
                            batchIndex, batch.size(), results.size());
                    logMemory("text-det:after-batch-" + batchIndex);
                } finally {
                    for (PreProcessResult input : inputs) {
                        releasePreProcessResult(matManager, input);
                    }
                    for (Mat sourceImage : sourceImages) {
                        matManager.release(sourceImage);
                    }
                }
                batchIndex++;
            }
        }
        logMemory("text-det:end");
        System.out.println("Saved line crops: " + lineDir.toAbsolutePath());
    }

    private static void runTextRecognition(Path outDir, int batchSize) throws Exception {
        List<Path> images = listPngs(outDir.resolve("line"));
        System.out.printf("Text line crops: %d, batchSize=%d%n", images.size(), batchSize);
        logMemory("text-rec:start");
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextRecognitionModel model = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, GPU_INDEX);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            logMemory("text-rec:after-model-load");
            int batchIndex = 0;
            for (List<Path> batch : split(images, batchSize)) {
                List<PreProcessResult> inputs = new ArrayList<>();
                try {
                    for (Path image : batch) {
                        inputs.add(model.processRgb(matManager, ImageUtil.readToRgb(matManager, image.toString()), ndManager));
                    }
                    logMemory("text-rec:before-batch-" + batchIndex);
                    List<List<RecognitionResult>> results = model.batchPredict(inputs, batchSize, matManager, ndManager, Map.of());
                    System.out.printf("Text-rec batch %d done: inputs=%d, results=%d%n",
                            batchIndex, batch.size(), results.size());
                    logMemory("text-rec:after-batch-" + batchIndex);
                } finally {
                    for (PreProcessResult input : inputs) {
                        releasePreProcessResult(matManager, input);
                    }
                }
                batchIndex++;
            }
        }
        logMemory("text-rec:end");
    }

    private static void saveLineCrops(List<Path> batch,
                                      List<Mat> sourceImages,
                                      List<TextDetectionResult> results,
                                      Path lineDir,
                                      MatManager matManager,
                                      NDManager ndManager) {
        for (int i = 0; i < results.size() && i < sourceImages.size(); i++) {
            TextDetectionResult result = results.get(i);
            if (result == null || result.polys() == null) {
                continue;
            }
            int lineIndex = 0;
            for (int[][] poly : result.polys()) {
                Mat line = ImageUtil.getMinAreaRectCrop(matManager, ndManager, sourceImages.get(i), poly);
                try {
                    String stem = stripExtension(batch.get(i).getFileName().toString());
                    Path linePath = lineDir.resolve("%s_line_%03d.png".formatted(stem, lineIndex));
                    Imgcodecs.imwrite(linePath.toString(), line);
                } finally {
                    matManager.release(line);
                }
                lineIndex++;
            }
        }
    }

    private static Mat cropRegion(MatManager matManager, Mat srcImage, float[] coordinate) {
        int x1 = Math.max(0, Math.round(coordinate[0]));
        int y1 = Math.max(0, Math.round(coordinate[1]));
        int x2 = Math.min(srcImage.cols(), Math.round(coordinate[2]));
        int y2 = Math.min(srcImage.rows(), Math.round(coordinate[3]));
        Rect rect = new Rect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
        Mat roi = matManager.newMat(srcImage, rect);
        try {
            return matManager.cloneMat(roi);
        } finally {
            matManager.release(roi);
        }
    }

    private static void releasePreProcessResult(MatManager matManager, PreProcessResult ppr) {
        if (ppr == null) {
            return;
        }
        IOUtil.close(ppr.ndArray());
        matManager.release(ppr.mat());
    }

    private static List<Path> listPngs(Path dir) throws Exception {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> paths = Files.list(dir)) {
            return paths
                    .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".png"))
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
    }

    private static <T> List<List<T>> split(List<T> values, int batchSize) {
        if (values.isEmpty()) {
            return List.of();
        }
        List<List<T>> batches = new ArrayList<>();
        for (int start = 0; start < values.size(); start += batchSize) {
            batches.add(values.subList(start, Math.min(start + batchSize, values.size())));
        }
        return batches;
    }

    private static int parseBatchSize(String[] args, int index, int defaultValue) {
        if (args.length <= index) {
            return defaultValue;
        }
        return Math.max(1, Integer.parseInt(args[index]));
    }

    private static String sanitize(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        return value.replaceAll("[^A-Za-z0-9_.-]+", "_");
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? name : name.substring(0, dot);
    }

    private static void logMemory(String label) {
        Runtime runtime = Runtime.getRuntime();
        double heapUsedGb = (runtime.totalMemory() - runtime.freeMemory()) / 1024d / 1024d / 1024d;
        System.out.printf("MEMORY %-28s heapUsedGB=%.2f, %s%n",
                label, heapUsedGb, windowsProcessMemory());
    }

    private static String windowsProcessMemory() {
        long pid = ProcessHandle.current().pid();
        String command = "$p=Get-Process -Id " + pid
                + "; 'workingSetGB={0:N2}, privateGB={1:N2}' -f ($p.WorkingSet64/1GB),($p.PrivateMemorySize64/1GB)";
        try {
            Process process = new ProcessBuilder(
                    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)
                    .redirectErrorStream(true)
                    .start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            int exit = process.waitFor();
            if (exit == 0 && !output.isBlank()) {
                return output;
            }
            if (!output.isBlank()) {
                return "workingSetGB=unknown, privateGB=unknown, psExit=" + exit
                        + ", psOutput=" + output.replaceAll("\\s+", " ");
            }
        } catch (Exception ignored) {
            // Fall through to a portable-but-less-useful marker.
        }
        return "workingSetGB=unknown, privateGB=unknown";
    }

    private static void usage() {
        System.out.println("""
                Usage:
                  extract <pdfPath> <pages> <outDir>
                  formula <outDir> [batchSize]
                  table <outDir> [batchSize]
                  text-det <outDir> [batchSize]
                  text-rec <outDir> [batchSize]

                Example:
                  extract E:\\flux-data\\2606.13108_zh_CN.pdf 9-12 target\\ocr-memory-probe\\2606_zh_9_12
                  formula target\\ocr-memory-probe\\2606_zh_9_12 4
                """);
    }
}
