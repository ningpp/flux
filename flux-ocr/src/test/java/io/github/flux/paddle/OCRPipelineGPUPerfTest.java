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
import io.github.flux.util.ImageUtil;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.Mat;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * OCRPipeline GPU Performance Benchmark
 *
 * Models:
 *   det          --> PP-OCRv6_medium_det
 *   rec          --> PP-OCRv6_medium_rec
 *   doc_ori      --> PP-LCNet_x1_0_doc_ori
 *   textline_ori --> PP-LCNet_x1_0_textline_ori
 *   layout       --> PP-DocLayoutV3
 *   table        --> unirec-0.1b
 *   formula      --> pix2text-mfr-1.5
 *
 * PDF-to-Image: PDFBox, DPI=300
 * Inference: GPU (gpuIndex=0)
 */
public class OCRPipelineGPUPerfTest {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final int DPI = 300;
    private static final int GPU_INDEX = 0;
    private static final int TABLE_DECODER_GPU_INDEX = -1;
    private static final int TABLE_MAX_TOKENS = 768;

    private static final String[] PDF_FILES = {
        // https://hjfy.top/arxiv/2606.13108
        "E:\\flux-data\\2606.13108_zh_CN.pdf",
        "E:\\flux-data\\2606.13108.pdf",
        // https://hjfy.top/arxiv/2606.13392
        "E:\\flux-data\\2606.13392_zh_CN.pdf",
        "E:\\flux-data\\2606.13392.pdf"
    };

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        int iterations = 1;
        int pipelinePageBatchSize = 4;
        int layoutBatchSize = 1;
        int detectionBatchSize = 1;
        int recognitionBatchSize = 8;
        int formulaBatchSize = 4;
        // Verify PDF files exist
        for (String pdfFile : PDF_FILES) {
            File f = new File(pdfFile);
            if (!f.exists()) {
                System.err.println("PDF file not found: " + pdfFile);
                return;
            }
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, GPU_INDEX);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, GPU_INDEX);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, GPU_INDEX);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_textline_ori", env, GPU_INDEX);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, "PP-DocLayoutV3", GPU_INDEX, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, "pix2text-mfr-1.5", GPU_INDEX, env);
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env,
                     Map.of(
                             "unirec.decoderGpuIndex", TABLE_DECODER_GPU_INDEX,
                             "unirec.maxTokens", TABLE_MAX_TOKENS
                     ))) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel,
                    docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);
            logMemory("after-model-load");

            Map<String, Object> params = new HashMap<>();
            params.put("layoutBatchSize", layoutBatchSize);
            params.put("detectionBatchSize", detectionBatchSize);
            params.put("recognitionBatchSize", recognitionBatchSize);
            params.put("formulaBatchSize", formulaBatchSize);
            params.put("tableBatchSize", 1);

            // Warm up with first page of first PDF
            System.out.println("========== Warm Up ==========");
            warmUp(pipeline, params);
            logMemory("after-warmup");

            // Run benchmark
            System.out.println("\n========== GPU Performance Benchmark ==========");
            System.out.println("DPI: " + DPI);
            System.out.println("GPU Index: " + GPU_INDEX);
            System.out.println("Pipeline page batch size: " + pipelinePageBatchSize);
            System.out.println("Layout batch size: " + layoutBatchSize);
            System.out.println("Detection batch size: " + detectionBatchSize);
            System.out.println("Recognition batch size: " + recognitionBatchSize);
            System.out.println("Formula batch size: " + formulaBatchSize);
            System.out.println("Table decoder GPU Index: " + TABLE_DECODER_GPU_INDEX);
            System.out.println("Table max tokens: " + TABLE_MAX_TOKENS);
            System.out.println();

            long totalE2ENanos = 0;       // end-to-end (PDF+OCR)
            long totalOcrOnlyNanos = 0;    // OCR only (exclude PDF-to-image)
            int totalPages = 0;
            int totalSuccessPages = 0;

            Map<String, Long> fileE2ENanos = new LinkedHashMap<>();
            Map<String, Long> fileOcrNanos = new LinkedHashMap<>();
            Map<String, Integer> filePages = new LinkedHashMap<>();
            Map<String, Integer> fileSuccessPages = new LinkedHashMap<>();

            for (int iter = 0; iter < iterations; iter++) {

                for (String pdfFile : PDF_FILES) {
                    String fileName = new File(pdfFile).getName();
                    System.out.println("--- Processing: " + fileName + " ---");

                    List<String> imagePaths = new ArrayList<>();
                    try {
                        // Convert PDF pages to images (timed separately)
                        LocalDateTime pdfConvertStart = LocalDateTime.now();
                        imagePaths = convertPdfToImages(pdfFile);
                        LocalDateTime pdfConvertEnd = LocalDateTime.now();
                        long pdfConvertNanos = Duration.between(pdfConvertStart, pdfConvertEnd).toNanos();
                        int pageCount = imagePaths.size();
                        System.out.println("  Pages: " + pageCount);
                        System.out.printf("  PDF-to-Image total: %.2f ms, avg %.2f ms/page%n",
                            pdfConvertNanos / 1000_000d, pdfConvertNanos / 1000_000d / (double) pageCount);

                        long fileOcrNanosSum = 0;
                        int successCount = 0;

                        // Process pages in batches so the pipeline benchmark exercises cross-page batching.
                        for (int batchStart = 0; batchStart < imagePaths.size(); batchStart += pipelinePageBatchSize) {
                            int batchEnd = Math.min(batchStart + pipelinePageBatchSize, imagePaths.size());
                            List<String> pageBatch = imagePaths.subList(batchStart, batchEnd);
                            try {
                                LocalDateTime batchOcrStart = LocalDateTime.now();
                                List<List<OCRPipelineResult>> results = pipeline.predict(pageBatch, params);
                                LocalDateTime batchOcrEnd = LocalDateTime.now();

                                long batchNanos = Duration.between(batchOcrStart, batchOcrEnd).toNanos();
                                if (results.size() != pageBatch.size()) {
                                    throw new IllegalStateException("Expected " + pageBatch.size()
                                            + " page results but got " + results.size());
                                }
                                fileOcrNanosSum += batchNanos;
                                successCount += results.size();
                                double batchAvgMs = batchNanos / 1000_000d / (double) pageBatch.size();

                                System.out.printf("  Batch pages %2d-%2d: OCR %8.2f ms, avg %7.2f ms/page%n",
                                        batchStart + 1, batchEnd, batchNanos / 1000_000d, batchAvgMs);
                                logMemory("after-batch-" + fileName + "-" + (batchStart + 1) + "-" + batchEnd);

                                // Print brief result summary
                                for (int i = 0; i < results.size(); i++) {
                                    List<OCRPipelineResult> pageResults = results.get(i);
                                    int regionCount = 0;
                                    int textLineCount = 0;
                                    int formulaCount = 0;
                                    int tableCount = 0;
                                    for (OCRPipelineResult r : pageResults) {
                                        if (r.layoutRegions() != null) {
                                            for (LayoutRegionResult region : r.layoutRegions()) {
                                                regionCount++;
                                                switch (region.regionType()) {
                                                    case "text" ->
                                                        textLineCount += region.textResults() != null ? region.textResults().size() : 0;
                                                    case "formula" -> formulaCount++;
                                                    case "table" -> tableCount++;
                                                }
                                            }
                                        }
                                    }
                                    System.out.printf("    Page %2d: %d regions (text:%d, formula:%d, table:%d)%n",
                                            batchStart + i + 1, regionCount, textLineCount, formulaCount, tableCount);
                                }
                            } catch (Exception e) {
                                System.out.printf("  Batch pages %2d-%2d: FAILED - %s%n",
                                        batchStart + 1, batchEnd, e.getMessage());
                                throw e;
                            }
                        }

                        long fileE2ENanosSum = pdfConvertNanos + fileOcrNanosSum;
                        double fileAvgE2EMs = fileE2ENanosSum / 1000_000d / (double) pageCount;
                        double fileAvgOcrMs = successCount == 0 ? 0 : fileOcrNanosSum / 1000_000d / (double) successCount;

                        totalE2ENanos += fileE2ENanosSum;
                        totalOcrOnlyNanos += fileOcrNanosSum;
                        totalPages += pageCount;
                        totalSuccessPages += successCount;

                        fileE2ENanos.put(fileName, fileE2ENanosSum);
                        fileOcrNanos.put(fileName, fileOcrNanosSum);
                        filePages.put(fileName, pageCount);
                        fileSuccessPages.put(fileName, successCount);

                        System.out.printf("  OCR total: %.2f ms%n", fileOcrNanosSum / 1000_000d);
                        System.out.printf("  E2E avg per page (PDF+OCR): %.2f ms%n", fileAvgE2EMs);
                        System.out.printf("  OCR avg per page (exclude PDF): %.2f ms%n", fileAvgOcrMs);
                        logMemory("after-file-" + fileName);
                    } finally {
                        // Clean up temp images even if OCR fails mid-file
                        for (String imagePath : imagePaths) {
                            new File(imagePath).delete();
                        }
                    }
                }
            }
            // Summary
            double overallAvgE2EMs = totalE2ENanos / 1000_000d / (double) totalPages;
            double overallAvgOcrMs = totalSuccessPages == 0 ? 0 : totalOcrOnlyNanos / 1000_000d / (double) totalSuccessPages;
            System.out.println("\n========== Performance Summary ==========");
            System.out.printf("Total iterations: %d%n", iterations);
            System.out.printf("Total pages: %d (success: %d)%n", totalPages, totalSuccessPages);
            System.out.printf("E2E avg per page (PDF+OCR): %.2f ms%n", overallAvgE2EMs);
            System.out.printf("OCR avg per page (exclude PDF): %.2f ms%n", overallAvgOcrMs);
            System.out.printf("Peak working set/private: %.2f GB / %.2f GB%n",
                    peakWorkingSetGb, peakPrivateGb);
            System.out.println();
            System.out.println("Per-file details:");
            System.out.printf("  %-30s %6s %8s %8s %8s%n", "File", "Pages", "E2E(ms)", "OCR(ms)", "PDF(ms)");
            List<String> fileNames = fileE2ENanos.keySet().stream().toList().stream().sorted().toList();
            for (String fileName : fileNames) {
                int pages = filePages.get(fileName);
                int success = fileSuccessPages.get(fileName);
                long e2e = fileE2ENanos.get(fileName);
                long ocr = fileOcrNanos.get(fileName);
                long pdf = e2e - ocr;
                double ocrAvg = success == 0 ? 0 : ocr / 1000_000d / (double) success;
                System.out.printf("  %-30s %6d %8.2f %8.2f %8.2f%n",
                        fileName, pages,
                        e2e / 1000_000d / (double) pages,
                        ocrAvg,
                        pdf / 1000_000d / (double) pages);
            }
        }
    }

    private static void warmUp(OCRPipeline pipeline, Map<String, Object> params) throws Exception {
        try (PDDocument document = Loader.loadPDF(new File(PDF_FILES[0]));
             MatManager matManager = new MatManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage bufferedImage = renderer.renderImageWithDPI(0, DPI, ImageType.RGB);
            Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
            Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);

            String tempPath = System.getProperty("java.io.tmpdir") + File.separator + "flux_warmup.png";
            org.opencv.imgcodecs.Imgcodecs.imwrite(tempPath, rgbMat);
            matManager.release(mat);
            matManager.release(rgbMat);

            pipeline.predict(List.of(tempPath), params);

            new File(tempPath).delete();
            System.out.println("Warm up completed.");
        }
    }

    private static List<String> convertPdfToImages(String pdfFile) throws Exception {
        List<String> imagePaths = new ArrayList<>();
        try (PDDocument document = Loader.loadPDF(new File(pdfFile));
             MatManager matManager = new MatManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            int pageCount = document.getNumberOfPages();
            String baseName = new File(pdfFile).getName().replace(".pdf", "");

            for (int i = 0; i < pageCount; i++) {
                BufferedImage bufferedImage = renderer.renderImageWithDPI(i, DPI, ImageType.RGB);
                Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
                Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);

                String imagePath = System.getProperty("java.io.tmpdir") + File.separator
                        + baseName + "_page_" + (i + 1) + ".png";
                org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, rgbMat);
                imagePaths.add(imagePath);
                matManager.release(mat);
                matManager.release(rgbMat);
            }
        }
        return imagePaths;
    }

    private static double peakWorkingSetGb = 0d;
    private static double peakPrivateGb = 0d;

    private static void logMemory(String label) {
        Runtime runtime = Runtime.getRuntime();
        double heapUsedGb = (runtime.totalMemory() - runtime.freeMemory()) / 1024d / 1024d / 1024d;
        ProcessMemorySnapshot snapshot = windowsProcessMemory();
        peakWorkingSetGb = Math.max(peakWorkingSetGb, snapshot.workingSetGb());
        peakPrivateGb = Math.max(peakPrivateGb, snapshot.privateGb());
        System.out.printf("  MEMORY %-36s heapUsedGB=%.2f, workingSetGB=%.2f, privateGB=%.2f%n",
                label, heapUsedGb, snapshot.workingSetGb(), snapshot.privateGb());
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
            // Keep benchmark running even if memory sampling is unavailable.
        }
        return new ProcessMemorySnapshot(0d, 0d);
    }

    private record ProcessMemorySnapshot(double workingSetGb, double privateGb) {
    }
}
