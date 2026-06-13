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

        int iterations = 10;
        int recognitionBatchSize = 2;
        int formulaBatchSize = 2;
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
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env)) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel,
                    docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);

            Map<String, Object> params = new HashMap<>();
            params.put("recognitionBatchSize", recognitionBatchSize);
            params.put("formulaBatchSize", formulaBatchSize);

            // Warm up with first page of first PDF
            System.out.println("========== Warm Up ==========");
            warmUp(pipeline, params);

            // Run benchmark
            System.out.println("\n========== GPU Performance Benchmark ==========");
            System.out.println("DPI: " + DPI);
            System.out.println("GPU Index: " + GPU_INDEX);
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

                    // Convert PDF pages to images (timed separately)
                    LocalDateTime pdfConvertStart = LocalDateTime.now();
                    List<String> imagePaths = convertPdfToImages(pdfFile);
                    LocalDateTime pdfConvertEnd = LocalDateTime.now();
                    long pdfConvertNanos = Duration.between(pdfConvertStart, pdfConvertEnd).toNanos();
                    int pageCount = imagePaths.size();
                    System.out.println("  Pages: " + pageCount);
                    System.out.printf("  PDF-to-Image total: %.2f ms, avg %.2f ms/page%n",
                        pdfConvertNanos / 1000_000d, pdfConvertNanos / 1000_000d / (double) pageCount);

                    long fileOcrNanosSum = 0;
                    int successCount = 0;

                    // Process pages one by one for accurate per-page timing
                    for (int i = 0; i < imagePaths.size(); i++) {
                        String imagePath = imagePaths.get(i);
                        try {
                            LocalDateTime pageStart = LocalDateTime.now();
                            List<List<OCRPipelineResult>> results = pipeline.predict(List.of(imagePath), params);
                            LocalDateTime pageEnd = LocalDateTime.now();

                            long pageNanos = Duration.between(pageStart, pageEnd).toNanos();
                            fileOcrNanosSum += pageNanos;
                            successCount++;

                            // Print brief result summary
                            List<OCRPipelineResult> pageResults = results.get(0);
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
                            System.out.printf("  Page %2d: OCR %7.2f ms, %d regions (text:%d, formula:%d, table:%d)%n",
                                i + 1, pageNanos / 1000_000d, regionCount, textLineCount, formulaCount, tableCount);
                        } catch (Exception e) {
                            System.out.printf("  Page %2d: FAILED - %s%n", i + 1, e.getMessage());
                        }
                    }

                    long fileE2ENanosSum = pdfConvertNanos + fileOcrNanosSum;
                    double fileAvgE2EMs = fileE2ENanosSum / 1000_000d / (double) pageCount;
                    double fileAvgOcrMs = fileOcrNanosSum / 1000_000d / (double) pageCount;

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

                    // Clean up temp images
                    for (String imagePath : imagePaths) {
                        new File(imagePath).delete();
                    }
                }
            }
            // Summary
            double overallAvgE2EMs = totalE2ENanos / 1000_000d / (double) totalPages;
            double overallAvgOcrMs = totalOcrOnlyNanos / 1000_000d / (double) totalSuccessPages;
            System.out.println("\n========== Performance Summary ==========");
            System.out.printf("Total iterations: %d%n", iterations);
            System.out.printf("Total pages: %d (success: %d)%n", totalPages, totalSuccessPages);
            System.out.printf("E2E avg per page (PDF+OCR): %.2f ms%n", overallAvgE2EMs);
            System.out.printf("OCR avg per page (exclude PDF): %.2f ms%n", overallAvgOcrMs);
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
                System.out.printf("  %-30s %6d %8.2f %8.2f %8.2f%n",
                        fileName, pages,
                        e2e / 1000_000d / (double) pages,
                        ocr / 1000_000d / (double) success,
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
            }
        }
        return imagePaths;
    }
}
