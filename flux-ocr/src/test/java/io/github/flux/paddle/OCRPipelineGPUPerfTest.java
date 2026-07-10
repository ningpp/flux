package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.bytedeco.OpenCVImage;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.TextDetectionResult;
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
import io.github.flux.util.IOUtil;
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
 * OCRPipeline GPU Performance Benchmark with per-stage memory attribution
 * and per-model isolation diagnostics.
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
 * PDF-to-Image: PDFBox, DPI=300 (converted once, reused across iterations)
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

        int iterations = 1;
        int pipelinePageBatchSize = 1;
        int layoutBatchSize = 1;
        int docOrientationBatchSize = 1;
        int textLineOrientationBatchSize = 1;
        int detectionBatchSize = 1;
        int recognitionBatchSize = 1;
        int formulaBatchSize = 1;
        int tableBatchSize = 1;
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
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", GPU_INDEX, env, Map.of())) {

            // ===== Phase 1: Convert all PDFs to images ONCE =====
            System.out.println("========== PDF-to-Image Conversion (once) ==========");
            Map<String, List<String>> pdfImagePaths = new LinkedHashMap<>();
            Map<String, Long> pdfConvertNanosMap = new LinkedHashMap<>();
            for (String pdfFile : PDF_FILES) {
                String fileName = new File(pdfFile).getName();
                LocalDateTime pdfConvertStart = LocalDateTime.now();
                List<String> imagePaths = convertPdfToImages(pdfFile);
                LocalDateTime pdfConvertEnd = LocalDateTime.now();
                long pdfConvertNanos = Duration.between(pdfConvertStart, pdfConvertEnd).toNanos();
                pdfImagePaths.put(pdfFile, imagePaths);
                pdfConvertNanosMap.put(pdfFile, pdfConvertNanos);
                System.out.printf("  %s: %d pages, %.2f ms, avg %.2f ms/page%n",
                        fileName, imagePaths.size(),
                        pdfConvertNanos / 1000_000d,
                        pdfConvertNanos / 1000_000d / (double) imagePaths.size());
            }
            logMemory("after-pdf-convert");

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel,
                    docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);
            logMemory("after-model-load");

            Map<String, Object> params = new HashMap<>();
            params.put("layoutBatchSize", layoutBatchSize);
            params.put("docOrientationBatchSize", docOrientationBatchSize);
            params.put("textLineOrientationBatchSize", textLineOrientationBatchSize);
            params.put("detectionBatchSize", detectionBatchSize);
            params.put("recognitionBatchSize", recognitionBatchSize);
            params.put("formulaBatchSize", formulaBatchSize);
            params.put("tableBatchSize", tableBatchSize);

            // Warm up with first page of first PDF
            System.out.println("\n========== Warm Up ==========");
            List<String> warmUpImages = pdfImagePaths.get(PDF_FILES[0]);
            if (warmUpImages != null && !warmUpImages.isEmpty()) {
                pipeline.predictV2(warmUpImages.subList(0, 1), params);
                System.out.println("Warm up completed.");
            }
            logMemory("after-warmup");

            // ===== Phase 2: Main benchmark with per-stage memory tracking =====
            System.out.println("\n========== GPU Performance Benchmark ==========");
            System.out.println("DPI: " + DPI);
            System.out.println("GPU Index: " + GPU_INDEX);
            System.out.println("Pipeline page batch size: " + pipelinePageBatchSize);
            System.out.println("Layout batch size: " + layoutBatchSize);
            System.out.println("Doc orientation batch size: " + docOrientationBatchSize);
            System.out.println("Text line orientation batch size: " + textLineOrientationBatchSize);
            System.out.println("Detection batch size: " + detectionBatchSize);
            System.out.println("Recognition batch size: " + recognitionBatchSize);
            System.out.println("Formula batch size: " + formulaBatchSize);
            System.out.println("Table batch size: " + tableBatchSize);
            System.out.println();

            long totalOcrOnlyNanos = 0;
            int totalPages = 0;
            int totalSuccessPages = 0;

            Map<String, Long> fileOcrNanos = new LinkedHashMap<>();
            Map<String, Integer> filePages = new LinkedHashMap<>();
            Map<String, Integer> fileSuccessPages = new LinkedHashMap<>();

            List<MemoryPoint> allStageMemoryPoints = new ArrayList<>();

            // Wire memoryObserver for per-stage tracking
            params.put("memoryObserver", (Consumer<String>) stage -> {
                MemoryPoint point = MemoryPoint.capture("stage:" + stage, null);
                allStageMemoryPoints.add(point);
                System.out.printf(Locale.ROOT,
                        "  STAGE %-42s heapGB=%.2f workGB=%.2f privGB=%.2f%n",
                        stage, point.heapGb(), point.workingSetGb(), point.privateGb());
            });

            for (int iter = 0; iter < iterations; iter++) {
                for (String pdfFile : PDF_FILES) {
                    String fileName = new File(pdfFile).getName();
                    List<String> imagePaths = pdfImagePaths.get(pdfFile);
                    int pageCount = imagePaths.size();
                    System.out.println("--- Processing: " + fileName + " (iter " + (iter + 1) + ") ---");
                    System.out.println("  Pages: " + pageCount);

                    long fileOcrNanosSum = 0;
                    int successCount = 0;

                    for (int batchStart = 0; batchStart < imagePaths.size(); batchStart += pipelinePageBatchSize) {
                        int batchEnd = Math.min(batchStart + pipelinePageBatchSize, imagePaths.size());
                        List<String> pageBatch = imagePaths.subList(batchStart, batchEnd);
                        try {
                            LocalDateTime batchOcrStart = LocalDateTime.now();
                            List<List<OCRPipelineResult>> results = pipeline.predictV2(pageBatch, params);
                            LocalDateTime batchOcrEnd = LocalDateTime.now();

                            long batchNanos = Duration.between(batchOcrStart, batchOcrEnd).toNanos();
                            if (results.size() != pageBatch.size()) {
                                throw new IllegalStateException("Expected " + pageBatch.size()
                                        + " page results but got " + results.size());
                            }
                            fileOcrNanosSum += batchNanos;
                            successCount += results.size();
                            double batchAvgMs = batchNanos / 1000_000d / (double) pageBatch.size();

                            System.out.printf(Locale.ROOT, "  Batch pages %2d-%2d: OCR %8.2f ms, avg %7.2f ms/page%n",
                                    batchStart + 1, batchEnd, batchNanos / 1000_000d, batchAvgMs);
                            logMemory("after-batch-" + fileName + "-" + (batchStart + 1) + "-" + batchEnd);

                            printResultSummary(results, batchStart);
                        } catch (Exception e) {
                            System.out.printf("  Batch pages %2d-%2d: FAILED - %s%n",
                                    batchStart + 1, batchEnd, e.getMessage());
                            throw e;
                        }
                    }

                    totalOcrOnlyNanos += fileOcrNanosSum;
                    totalPages += pageCount;
                    totalSuccessPages += successCount;

                    fileOcrNanos.put(fileName, fileOcrNanosSum);
                    filePages.put(fileName, pageCount);
                    fileSuccessPages.put(fileName, successCount);

                    double fileAvgOcrMs = successCount == 0 ? 0 : fileOcrNanosSum / 1000_000d / (double) successCount;
                    System.out.printf(Locale.ROOT, "  OCR total: %.2f ms%n", fileOcrNanosSum / 1000_000d);
                    System.out.printf(Locale.ROOT, "  OCR avg per page: %.2f ms%n", fileAvgOcrMs);
                    logMemory("after-file-" + fileName);
                }
            }

            // Print per-stage memory attribution
            printStageMemoryDeltas(allStageMemoryPoints);

            // Summary
            double overallAvgOcrMs = totalSuccessPages == 0 ? 0 : totalOcrOnlyNanos / 1000_000d / (double) totalSuccessPages;
            System.out.println("\n========== Performance Summary ==========");
            System.out.printf("Total iterations: %d%n", iterations);
            System.out.printf("Total pages: %d (success: %d)%n", totalPages, totalSuccessPages);
            System.out.printf("OCR avg per page: %.2f ms%n", overallAvgOcrMs);
            System.out.printf("Peak working set/private: %.2f GB / %.2f GB%n",
                    peakWorkingSetGb, peakPrivateGb);
            System.out.println();
            System.out.println("Per-file details:");
            System.out.printf(Locale.ROOT, "  %-30s %6s %8s %8s%n", "File", "Pages", "OCR(ms)", "PDF(ms)");
            List<String> fileNames = fileOcrNanos.keySet().stream().toList().stream().sorted().toList();
            for (String fileName : fileNames) {
                int pages = filePages.get(fileName);
                int success = fileSuccessPages.get(fileName);
                long ocr = fileOcrNanos.get(fileName);
                // Find the PDF path that matches this fileName
                long pdfNanos = 0;
                for (int fi = 0; fi < PDF_FILES.length; fi++) {
                    if (new File(PDF_FILES[fi]).getName().equals(fileName)) {
                        pdfNanos = pdfConvertNanosMap.getOrDefault(PDF_FILES[fi], 0L);
                        break;
                    }
                }
                double ocrAvg = success == 0 ? 0 : ocr / 1000_000d / (double) success;
                System.out.printf(Locale.ROOT, "  %-30s %6d %8.2f %8.2f%n",
                        fileName, pages,
                        ocrAvg,
                        pdfNanos / 1000_000d / (double) pages);
            }

            // ===== Phase 4: Per-model isolation benchmark =====
            System.out.println("\n========== Per-Model Isolation Benchmark ==========");
            List<ModelMemoryAttribution> modelAttributions = runPerModelIsolationBenchmark(
                    pdfImagePaths, detModel, recModel, docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel,
                    layoutBatchSize, detectionBatchSize, recognitionBatchSize, formulaBatchSize);
            printModelMemoryAttribution(modelAttributions);

            // ===== Phase 5: Root cause diagnosis =====
            System.out.println("\n========== Root Cause Diagnosis ==========");
            diagnoseRootCause(allStageMemoryPoints, modelAttributions);

            // Cleanup all temp images
            for (List<String> paths : pdfImagePaths.values()) {
                for (String path : paths) {
                    new File(path).delete();
                }
            }
        }
    }

    // ===== Per-model isolation benchmark =====

    private static List<ModelMemoryAttribution> runPerModelIsolationBenchmark(
            Map<String, List<String>> pdfImagePaths,
            TextDetectionModel detModel,
            TextRecognitionModel recModel,
            DocOrientationClassifyModel docOriModel,
            TextLineOrientationModel textLineOriModel,
            LayoutModel layoutModel,
            FormulaRecognitionModel formulaModel,
            TableModel tableModel,
            int layoutBatchSize,
            int detectionBatchSize,
            int recognitionBatchSize,
            int formulaBatchSize) throws Exception {

        List<ModelMemoryAttribution> attributions = new ArrayList<>();
        String tmpDir = System.getProperty("java.io.tmpdir") + File.separator;

        // Collect all page image paths across all PDFs
        List<String> allPagePaths = new ArrayList<>();
        for (List<String> paths : pdfImagePaths.values()) {
            allPagePaths.addAll(paths);
        }
        if (allPagePaths.isEmpty()) {
            return attributions;
        }

        // --- Layout isolation ---
        System.out.println("\n--- Layout Model Isolation ---");
        List<String> textCropPaths = new ArrayList<>();
        List<String> formulaCropPaths = new ArrayList<>();
        List<String> tableCropPaths = new ArrayList<>();
        {
            MemoryPoint before = MemoryPoint.capture("layout:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                List<List<ObjectDetectionResult>> allLayoutResults = new ArrayList<>();
                for (int batchStart = 0; batchStart < allPagePaths.size(); batchStart += layoutBatchSize) {
                    int batchEnd = Math.min(batchStart + layoutBatchSize, allPagePaths.size());
                    List<String> batch = allPagePaths.subList(batchStart, batchEnd);
                    List<ProcessedMat> layoutInputs = new ArrayList<>();
                    List<Mat> srcMats = new ArrayList<>();
                    List<Mat> srcRgbs = new ArrayList<>();
                    try {
                        for (String imgPath : batch) {
                            Mat src = matManager.imread(imgPath, Imgcodecs.IMREAD_COLOR_BGR);
                            srcMats.add(src);
                            Mat srcRgb = matManager.newMat();
                            Imgproc.cvtColor(src, srcRgb, Imgproc.COLOR_BGR2RGB);
                            srcRgbs.add(srcRgb);
                            layoutInputs.add(layoutModel.processRgb(matManager, srcRgb, ndManager));
                        }
                        List<List<ObjectDetectionResult>> batchResults = layoutModel.batchPredict(
                                layoutInputs, layoutBatchSize, matManager, ndManager, Map.of());
                        allLayoutResults.addAll(batchResults);

                        // Crop regions and save as images
                        for (int i = 0; i < batchResults.size(); i++) {
                            int pageIdx = batchStart + i;
                            Mat src = srcMats.get(i);
                            List<ObjectDetectionResult> regions = batchResults.get(i);
                            for (int r = 0; r < regions.size(); r++) {
                                ObjectDetectionResult region = regions.get(r);
                                Mat crop = cropRegion(matManager, src, region.coordinate());
                                Mat cropRgb = matManager.newMat();
                                Imgproc.cvtColor(crop, cropRgb, Imgproc.COLOR_BGR2RGB);
                                String cropType = classifyLabel(region.label());
                                String cropPath = tmpDir + "layout_crop_p" + pageIdx + "_r" + r + "_" + cropType + ".png";
                                Imgcodecs.imwrite(cropPath, cropRgb);
                                matManager.release(crop);
                                matManager.release(cropRgb);
                                switch (cropType) {
                                    case "formula" -> formulaCropPaths.add(cropPath);
                                    case "table" -> tableCropPaths.add(cropPath);
                                    default -> textCropPaths.add(cropPath);
                                }
                            }
                        }
                    } finally {
                        for (ProcessedMat input : layoutInputs) {
                            input.release(matManager);
                        }
                        for (Mat srcRgb : srcRgbs) {
                            matManager.release(srcRgb);
                        }
                        for (Mat src : srcMats) {
                            matManager.release(src);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("layout:after", null);
            attributions.add(new ModelMemoryAttribution("layout",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  layout: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, crops=%d (text=%d, formula=%d, table=%d)%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak,
                    textCropPaths.size() + formulaCropPaths.size() + tableCropPaths.size(),
                    textCropPaths.size(), formulaCropPaths.size(), tableCropPaths.size());
        }

        // --- Doc Orientation isolation ---
        System.out.println("\n--- Doc Orientation Model Isolation ---");
        {
            MemoryPoint before = MemoryPoint.capture("doc_ori:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (int batchStart = 0; batchStart < allPagePaths.size(); batchStart += layoutBatchSize) {
                    int batchEnd = Math.min(batchStart + layoutBatchSize, allPagePaths.size());
                    List<String> batch = allPagePaths.subList(batchStart, batchEnd);
                    List<PreProcessResult> docOriInputs = new ArrayList<>();
                    List<Mat> docOriRgbs = new ArrayList<>();
                    try {
                        for (String imgPath : batch) {
                            Mat bgr = matManager.imread(imgPath, Imgcodecs.IMREAD_COLOR_BGR);
                            Mat rgb = matManager.newMat();
                            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
                            matManager.release(bgr);
                            docOriRgbs.add(rgb);
                            docOriInputs.add(docOriModel.processRgb(matManager, rgb, ndManager));
                        }
                        docOriModel.batchPredict(docOriInputs, layoutBatchSize, matManager, ndManager, Map.of());
                    } finally {
                        for (PreProcessResult input : docOriInputs) {
                            matManager.release(input.mat());
                            IOUtil.close(input.ndArray());
                        }
                        for (Mat rgb : docOriRgbs) {
                            matManager.release(rgb);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("doc_ori:after", null);
            attributions.add(new ModelMemoryAttribution("doc-orientation",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  doc-orientation: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, closeablesPeak=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak, closeablesPeak);
        }

        // --- Text Detection isolation ---
        System.out.println("\n--- Text Detection Model Isolation ---");
        List<String> textLineCropPaths = new ArrayList<>();
        if (!textCropPaths.isEmpty()) {
            MemoryPoint before = MemoryPoint.capture("det:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (int batchStart = 0; batchStart < textCropPaths.size(); batchStart += detectionBatchSize) {
                    int batchEnd = Math.min(batchStart + detectionBatchSize, textCropPaths.size());
                    List<String> batch = textCropPaths.subList(batchStart, batchEnd);
                    List<PreProcessResult> detInputs = new ArrayList<>();
                    List<Mat> detMats = new ArrayList<>();
                    try {
                        for (String cropPath : batch) {
                            Mat bgr = matManager.imread(cropPath, Imgcodecs.IMREAD_COLOR_BGR);
                            detMats.add(bgr);
                            detInputs.add(new PreProcessResult(bgr, null));
                        }
                        List<TextDetectionResult> detResults = detModel.batchPredict(
                                detInputs, detectionBatchSize, matManager, ndManager, Map.of());
                        // Crop text lines and save
                        for (int i = 0; i < detResults.size(); i++) {
                            TextDetectionResult detResult = detResults.get(i);
                            Mat srcMat = detMats.get(i);
                            if (detResult.polys() != null) {
                                for (int l = 0; l < detResult.polys().length; l++) {
                                    try {
                                        Mat lineCrop = ImageUtil.getMinAreaRectCrop(matManager, ndManager, srcMat, detResult.polys()[l]);
                                        String linePath = tmpDir + "det_line_" + (batchStart + i) + "_l" + l + ".png";
                                        Imgcodecs.imwrite(linePath, lineCrop);
                                        textLineCropPaths.add(linePath);
                                        matManager.release(lineCrop);
                                    } catch (Exception e) {
                                        // Skip problematic line crops
                                    }
                                }
                            }
                        }
                    } finally {
                        for (PreProcessResult input : detInputs) {
                            matManager.release(input.mat());
                        }
                        for (Mat detMat : detMats) {
                            matManager.release(detMat);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("det:after", null);
            attributions.add(new ModelMemoryAttribution("text-detection",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  text-detection: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, lineCrops=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak, textLineCropPaths.size());
        }

        // --- Text Recognition isolation ---
        System.out.println("\n--- Text Recognition Model Isolation ---");
        if (!textLineCropPaths.isEmpty()) {
            MemoryPoint before = MemoryPoint.capture("rec:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (int batchStart = 0; batchStart < textLineCropPaths.size(); batchStart += recognitionBatchSize) {
                    int batchEnd = Math.min(batchStart + recognitionBatchSize, textLineCropPaths.size());
                    List<String> batch = textLineCropPaths.subList(batchStart, batchEnd);
                    List<PreProcessResult> recInputs = new ArrayList<>();
                    List<Mat> recMats = new ArrayList<>();
                    try {
                        for (String linePath : batch) {
                            Mat bgr = matManager.imread(linePath, Imgcodecs.IMREAD_COLOR_BGR);
                            recMats.add(bgr);
                            recInputs.add(new PreProcessResult(bgr, null));
                        }
                        recModel.batchPredict(recInputs, recognitionBatchSize, matManager, ndManager, Map.of());
                    } finally {
                        for (PreProcessResult input : recInputs) {
                            matManager.release(input.mat());
                        }
                        for (Mat recMat : recMats) {
                            matManager.release(recMat);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("rec:after", null);
            attributions.add(new ModelMemoryAttribution("text-recognition",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  text-recognition: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak);
        }

        // --- Text Line Orientation isolation ---
        System.out.println("\n--- Text Line Orientation Model Isolation ---");
        if (!textLineCropPaths.isEmpty()) {
            MemoryPoint before = MemoryPoint.capture("textline_ori:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (int batchStart = 0; batchStart < textLineCropPaths.size(); batchStart += recognitionBatchSize) {
                    int batchEnd = Math.min(batchStart + recognitionBatchSize, textLineCropPaths.size());
                    List<String> batch = textLineCropPaths.subList(batchStart, batchEnd);
                    List<PreProcessResult> lineOriInputs = new ArrayList<>();
                    List<Mat> lineOriRgbs = new ArrayList<>();
                    try {
                        for (String cropPath : batch) {
                            Mat bgr = matManager.imread(cropPath, Imgcodecs.IMREAD_COLOR_BGR);
                            Mat rgb = matManager.newMat();
                            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
                            matManager.release(bgr);
                            lineOriRgbs.add(rgb);
                            lineOriInputs.add(textLineOriModel.processRgb(matManager, rgb, ndManager));
                        }
                        textLineOriModel.batchPredict(lineOriInputs, recognitionBatchSize, matManager, ndManager, Map.of());
                    } finally {
                        for (PreProcessResult input : lineOriInputs) {
                            matManager.release(input.mat());
                            IOUtil.close(input.ndArray());
                        }
                        for (Mat rgb : lineOriRgbs) {
                            matManager.release(rgb);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("textline_ori:after", null);
            attributions.add(new ModelMemoryAttribution("textline-orientation",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  textline-orientation: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, closeablesPeak=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak, closeablesPeak);
        }

        // --- Formula Recognition isolation ---
        System.out.println("\n--- Formula Recognition Model Isolation ---");
        if (!formulaCropPaths.isEmpty()) {
            MemoryPoint before = MemoryPoint.capture("formula:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (int batchStart = 0; batchStart < formulaCropPaths.size(); batchStart += formulaBatchSize) {
                    int batchEnd = Math.min(batchStart + formulaBatchSize, formulaCropPaths.size());
                    List<String> batch = formulaCropPaths.subList(batchStart, batchEnd);
                    List<PreProcessResult> formulaInputs = new ArrayList<>();
                    List<Mat> formulaMats = new ArrayList<>();
                    List<Mat> formulaRgbs = new ArrayList<>();
                    try {
                        for (String cropPath : batch) {
                            Mat bgr = matManager.imread(cropPath, Imgcodecs.IMREAD_COLOR_BGR);
                            Mat rgb = matManager.newMat();
                            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
                            matManager.release(bgr);
                            formulaMats.add(bgr);
                            formulaRgbs.add(rgb);
                            formulaInputs.add(formulaModel.processRgb(matManager, rgb, ndManager));
                        }
                        formulaModel.batchPredict(formulaInputs, formulaBatchSize, matManager, ndManager, Map.of());
                    } finally {
                        for (PreProcessResult input : formulaInputs) {
                            matManager.release(input.mat());
                            IOUtil.close(input.ndArray());
                        }
                        for (Mat rgb : formulaRgbs) {
                            matManager.release(rgb);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("formula:after", null);
            attributions.add(new ModelMemoryAttribution("formula",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  formula: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, closeablesPeak=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak, closeablesPeak);
        }

        // --- Table Recognition isolation ---
        System.out.println("\n--- Table Recognition Model Isolation ---");
        if (!tableCropPaths.isEmpty()) {
            MemoryPoint before = MemoryPoint.capture("table:before", null);
            int matsPeak = 0;
            int closeablesPeak = 0;
            try (MatManager matManager = new MatManager();
                 NDManager ndManager = NDManager.newBaseManager()) {
                for (String cropPath : tableCropPaths) {
                    List<PreProcessResult> tableInputs = new ArrayList<>();
                    List<Mat> tableRgbs = new ArrayList<>();
                    try {
                        Mat bgr = matManager.imread(cropPath, Imgcodecs.IMREAD_COLOR_BGR);
                        Mat rgb = matManager.newMat();
                        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
                        matManager.release(bgr);
                        tableRgbs.add(rgb);
                        tableInputs.add(tableModel.processRgb(matManager, rgb, ndManager));
                        tableModel.batchPredict(tableInputs, 1, matManager, ndManager, Map.of());
                    } finally {
                        for (PreProcessResult input : tableInputs) {
                            matManager.release(input.mat());
                            IOUtil.close(input.ndArray());
                        }
                        for (Mat rgb : tableRgbs) {
                            matManager.release(rgb);
                        }
                    }
                    matsPeak = Math.max(matsPeak, matManager.trackedMatCount());
                    closeablesPeak = Math.max(closeablesPeak, matManager.trackedCloseableCount());
                }
            }
            forceGc();
            MemoryPoint after = MemoryPoint.capture("table:after", null);
            attributions.add(new ModelMemoryAttribution("table",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    after.workingSetGb() - before.workingSetGb(),
                    matsPeak, closeablesPeak));
            System.out.printf(Locale.ROOT, "  table: privDelta=%+.3fGB, heapDelta=%+.3fGB, matsPeak=%d, closeablesPeak=%d%n",
                    after.privateGb() - before.privateGb(),
                    after.heapGb() - before.heapGb(),
                    matsPeak, closeablesPeak);
        }

        // Cleanup isolation crop images
        List<String> allCropPaths = new ArrayList<>();
        allCropPaths.addAll(textCropPaths);
        allCropPaths.addAll(formulaCropPaths);
        allCropPaths.addAll(tableCropPaths);
        allCropPaths.addAll(textLineCropPaths);
        for (String cropPath : allCropPaths) {
            new File(cropPath).delete();
        }

        return attributions;
    }

    // ===== Reporting =====

    private static void printStageMemoryDeltas(List<MemoryPoint> points) {
        if (points.size() < 2) {
            return;
        }
        System.out.println("\n========== Per-Stage Memory Attribution ==========");
        System.out.printf(Locale.ROOT, "  %-42s %10s %10s%n", "Stage Transition", "PrivDeltaGB", "HeapDeltaGB");
        for (int i = 1; i < points.size(); i++) {
            MemoryPoint prev = points.get(i - 1);
            MemoryPoint curr = points.get(i);
            String transition = prev.label() + " -> " + curr.label().replace("stage:", "");
            System.out.printf(Locale.ROOT, "  %-42s %10.3f %10.3f%n",
                    transition,
                    curr.privateGb() - prev.privateGb(),
                    curr.heapGb() - prev.heapGb());
        }
    }

    private static void printModelMemoryAttribution(List<ModelMemoryAttribution> attributions) {
        if (attributions.isEmpty()) {
            return;
        }
        System.out.println("\n========== Per-Model Memory Attribution ==========");
        System.out.printf(Locale.ROOT, "  %-18s %12s %12s %12s %10s %12s%n",
                "Model", "PrivDeltaGB", "HeapDeltaGB", "WorkDeltaGB", "MatsPeak", "CloseablesPk");
        for (ModelMemoryAttribution attr : attributions) {
            System.out.printf(Locale.ROOT, "  %-18s %12.3f %12.3f %12.3f %10d %12d%n",
                    attr.modelName(), attr.privateDeltaGb(), attr.heapDeltaGb(),
                    attr.workingSetDeltaGb(), attr.matsPeak(), attr.closeablesPeak());
        }
    }

    private static void diagnoseRootCause(List<MemoryPoint> stagePoints, List<ModelMemoryAttribution> modelAttributions) {
        // Find pipeline stage with largest private memory growth
        String worstStage = "";
        double worstStageDelta = 0;
        for (int i = 1; i < stagePoints.size(); i++) {
            MemoryPoint prev = stagePoints.get(i - 1);
            MemoryPoint curr = stagePoints.get(i);
            double delta = curr.privateGb() - prev.privateGb();
            if (delta > worstStageDelta) {
                worstStageDelta = delta;
                worstStage = curr.label();
            }
        }

        // Find model with largest private memory growth in isolation
        String worstModel = "";
        double worstModelDelta = 0;
        for (ModelMemoryAttribution attr : modelAttributions) {
            if (attr.privateDeltaGb() > worstModelDelta) {
                worstModelDelta = attr.privateDeltaGb();
                worstModel = attr.modelName();
            }
        }

        // Check if predict:released actually releases memory
        double releasedDelta = 0;
        for (int i = 1; i < stagePoints.size(); i++) {
            if (stagePoints.get(i).label().contains("predict:released")) {
                releasedDelta = stagePoints.get(i).privateGb() - stagePoints.get(i - 1).privateGb();
                break;
            }
        }

        System.out.printf(Locale.ROOT, "  Worst pipeline stage (private growth): %s (%.3f GB)%n", worstStage, worstStageDelta);
        System.out.printf(Locale.ROOT, "  Worst isolated model (private growth):  %s (%.3f GB)%n", worstModel, worstModelDelta);
        System.out.printf(Locale.ROOT, "  predict:released private delta:          %.3f GB%n", releasedDelta);

        System.out.println();
        if (worstModelDelta > 0.1) {
            System.out.printf(Locale.ROOT, "  DIAGNOSIS: The '%s' model causes the most memory growth (%.3f GB private).%n", worstModel, worstModelDelta);
            switch (worstModel) {
                case "formula" ->
                    System.out.println("  ROOT CAUSE: Formula auto-regressive decoder (Pix2Text) holds encoderHiddenStates tensor across all decode steps. " +
                            "The ORT CUDA arena retains GPU memory for the peak allocation. Consider reducing formulaBatchSize or processing formulas sequentially.");
                case "table" ->
                    System.out.println("  ROOT CAUSE: Table auto-regressive decoder (Unirec) accumulates KV cache across decode steps. " +
                            "The ORT CUDA arena retains GPU memory for the peak allocation. Consider increasing GPU memory limit or processing tables sequentially.");
                case "text-detection" ->
                    System.out.println("  ROOT CAUSE: Text detection model creates padded batch images that vary in size per page, " +
                            "causing ORT CUDA arena to retain memory for the largest shape. Consider sorting pages by size before batching.");
                case "layout" ->
                    System.out.println("  ROOT CAUSE: Layout model (PP-DocLayoutV3) materializes order_logits float[batch][300][300] array on CPU heap. " +
                            "Consider reading OnnxTensor buffer directly instead of getValue().");
                case "doc-orientation" ->
                    System.out.println("  ROOT CAUSE: Doc orientation classifier (PP-LCNet_x1_0_doc_ori) is a lightweight CNN; " +
                            "private growth should be small. Verify it does not retain OnnxTensors/ORT Results across pages (closeablesPeak).");
                case "textline-orientation" ->
                    System.out.println("  ROOT CAUSE: Text-line orientation classifier (PP-LCNet_x1_0_textline_ori) is a lightweight CNN; " +
                            "private growth should be small. Verify it does not retain OnnxTensors/ORT Results across lines (closeablesPeak).");
                default ->
                    System.out.println("  ROOT CAUSE: Unknown. Check ORT CUDA arena configuration and per-session GPU memory limits.");
            }
        } else if (releasedDelta > -0.05) {
            System.out.println("  DIAGNOSIS: predict:released does NOT release significant private memory (" +
                    String.format(Locale.ROOT, "%.3f GB delta", releasedDelta) + "). Memory is retained by ORT CUDA arena across sessions.");
            System.out.println("  ROOT CAUSE: Cumulative ORT CUDA arena growth across multiple ONNX sessions. " +
                    "The arena allocator retains peak allocation for each session's lifetime. " +
                    "OnnxSessionUtil already disables memoryPatternOptimization and uses kSameAsRequested, " +
                    "but the arena still grows to peak and never shrinks within a session.");
        } else {
            System.out.println("  DIAGNOSIS: Memory is being released after predict (releasedDelta=" +
                    String.format(Locale.ROOT, "%.3f GB", releasedDelta) + "). No significant leak detected.");
        }

        // Print per-model closeables diagnostic
        System.out.println();
        for (ModelMemoryAttribution attr : modelAttributions) {
            if (attr.closeablesPeak() > 20) {
                System.out.printf(Locale.ROOT, "  WARNING: %s has high closeablesPeak=%d (OnnxTensors/ORT Results not released timely)%n",
                        attr.modelName(), attr.closeablesPeak());
            }
        }
    }

    // ===== Helpers =====

    private static Mat cropRegion(MatManager matManager, Mat srcImage, float[] coordinate) {
        if (coordinate == null) {
            return matManager.cloneMat(srcImage);
        }
        int x1 = Math.max(0, Math.round(coordinate[0]));
        int y1 = Math.max(0, Math.round(coordinate[1]));
        int x2 = Math.min(srcImage.cols(), Math.round(coordinate[2]));
        int y2 = Math.min(srcImage.rows(), Math.round(coordinate[3]));
        if (x2 <= x1 || y2 <= y1) {
            return matManager.cloneMat(srcImage);
        }
        Rect rect = new Rect(x1, y1, x2 - x1, y2 - y1);
        return matManager.newMat(srcImage, rect);
    }

    private static final java.util.Set<String> FORMULA_LABELS = java.util.Set.of(
            "display_formula", "inline_formula", "Formula");
    private static final java.util.Set<String> TABLE_LABELS = java.util.Set.of("table", "Table");
    private static final java.util.Set<String> IMAGE_LABELS = java.util.Set.of(
            "image", "chart", "seal", "header_image", "footer_image", "Picture");

    private static String classifyLabel(String label) {
        if (FORMULA_LABELS.contains(label)) return "formula";
        if (TABLE_LABELS.contains(label)) return "table";
        if (IMAGE_LABELS.contains(label)) return "image";
        return "text";
    }

    private static void printResultSummary(List<List<OCRPipelineResult>> results, int batchStart) {
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

    private static void forceGc() {
        System.gc();
        try {
            Thread.sleep(1500L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static double peakWorkingSetGb = 0d;
    private static double peakPrivateGb = 0d;

    private static void logMemory(String label) {
        Runtime runtime = Runtime.getRuntime();
        double heapUsedGb = (runtime.totalMemory() - runtime.freeMemory()) / 1024d / 1024d / 1024d;
        ProcessMemorySnapshot snapshot = windowsProcessMemory();
        peakWorkingSetGb = Math.max(peakWorkingSetGb, snapshot.workingSetGb());
        peakPrivateGb = Math.max(peakPrivateGb, snapshot.privateGb());
        System.out.printf(Locale.ROOT, "  MEMORY %-36s heapUsedGB=%.2f, workingSetGB=%.2f, privateGB=%.2f%n",
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

    private record MemoryPoint(String label, double heapGb, double workingSetGb, double privateGb) {
        static MemoryPoint capture(String label, MatManager matManager) {
            Runtime runtime = Runtime.getRuntime();
            double heapUsedGb = (runtime.totalMemory() - runtime.freeMemory()) / 1024d / 1024d / 1024d;
            ProcessMemorySnapshot process = windowsProcessMemory();
            return new MemoryPoint(label, heapUsedGb, process.workingSetGb(), process.privateGb());
        }
    }

    private record ModelMemoryAttribution(String modelName,
                                          double privateDeltaGb,
                                          double heapDeltaGb,
                                          double workingSetDeltaGb,
                                          int matsPeak,
                                          int closeablesPeak) {
    }
}
