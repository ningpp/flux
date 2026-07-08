package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.bytedeco.OpenCVImage;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
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
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Test-scope diagnostic runner for one PDF page.
 *
 * Usage:
 * mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineSinglePageDebug" "-Dexec.classpathScope=test" "-Dexec.args=\"full E:\\flux-data\\2606.13108_zh_CN.pdf 10 300 0\""
 * mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineSinglePageDebug" "-Dexec.classpathScope=test" "-Dexec.args=\"regions E:\\flux-data\\2606.13108_zh_CN.pdf 10 300 0\""
 */
public class OCRPipelineSinglePageDebug {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final String DEFAULT_PDF = "E:\\flux-data\\2606.13108_zh_CN.pdf";
    private static final int SAFE_TABLE_DECODER_GPU_INDEX = -1;
    private static final int SAFE_TABLE_MAX_TOKENS = 768;
    private static final Pattern RSS_PATTERN = Pattern.compile("([\\d,]+)\\s*K");

    private static final Set<String> FORMULA_LABELS = Set.of("display_formula", "inline_formula", "Formula");
    private static final Set<String> TABLE_LABELS = Set.of("table", "Table");

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String mode = args.length > 0 ? args[0] : "full";
        String pdfPath = args.length > 1 ? args[1] : DEFAULT_PDF;
        int pageNumber = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        int dpi = args.length > 3 ? Integer.parseInt(args[3]) : 300;
        int gpuIndex = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        boolean safeTable = mode.toLowerCase(Locale.ROOT).contains("safe");

        System.out.printf(Locale.ROOT,
                "mode=%s pdf=%s page=%d dpi=%d gpuIndex=%d safeTable=%s pid=%d at=%s%n",
                mode, pdfPath, pageNumber, dpi, gpuIndex, safeTable, ProcessHandle.current().pid(), LocalDateTime.now());
        if (safeTable) {
            System.out.printf(Locale.ROOT, "safe table config: decoderGpuIndex=%d maxTokens=%d%n",
                    SAFE_TABLE_DECODER_GPU_INDEX, SAFE_TABLE_MAX_TOKENS);
        }
        logMemory("process-start", gpuIndex, null);

        String imagePath = renderPage(pdfPath, pageNumber, dpi);
        try {
            logMemory("after-pdf-render", gpuIndex, null);
            if (mode.toLowerCase(Locale.ROOT).startsWith("regions")) {
                runRegionProbe(imagePath, gpuIndex, safeTable);
            } else {
                runFullPipeline(imagePath, gpuIndex, safeTable);
            }
        } finally {
            new File(imagePath).delete();
            logMemory("after-temp-delete", gpuIndex, null);
        }
    }

    private static void runFullPipeline(String imagePath, int gpuIndex, boolean safeTable) throws Exception {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, gpuIndex);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_textline_ori", env, gpuIndex);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, "PP-DocLayoutV3", gpuIndex, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, "pix2text-mfr-1.5", gpuIndex, env);
             TableModel tableModel = createTableModel(env, gpuIndex, safeTable)) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel, docOriModel, textLineOriModel, layoutModel, formulaModel, tableModel);

            Map<String, Object> params = new HashMap<>();
            params.put("recognitionBatchSize", 4);
            params.put("formulaBatchSize", 1);

            logMemory("full-after-model-load", gpuIndex, null);
            long t0 = System.nanoTime();
            try {
                List<List<OCRPipelineResult>> results = pipeline.predict(List.of(imagePath), params);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                List<OCRPipelineResult> pageResults = results.getFirst();
                int regionCount = 0;
                int formulaCount = 0;
                int tableCount = 0;
                int textLineCount = 0;
                for (OCRPipelineResult r : pageResults) {
                    if (r.layoutRegions() == null) {
                        continue;
                    }
                    for (LayoutRegionResult region : r.layoutRegions()) {
                        regionCount++;
                        if ("formula".equals(region.regionType())) {
                            formulaCount++;
                        } else if ("table".equals(region.regionType())) {
                            tableCount++;
                        } else if ("text".equals(region.regionType()) && region.textResults() != null) {
                            textLineCount += region.textResults().size();
                        }
                    }
                }
                System.out.printf(Locale.ROOT,
                        "FULL_RESULT ok ms=%d regions=%d textLines=%d formulas=%d tables=%d%n",
                        ms, regionCount, textLineCount, formulaCount, tableCount);
            } catch (Throwable t) {
                long ms = (System.nanoTime() - t0) / 1_000_000;
                System.out.printf(Locale.ROOT, "FULL_RESULT failed ms=%d error=%s%n", ms, t);
                t.printStackTrace(System.out);
            } finally {
                logMemory("full-after-predict", gpuIndex, null);
            }
        }
        logMemory("full-after-model-close", gpuIndex, null);
    }

    private static void runRegionProbe(String imagePath, int gpuIndex, boolean safeTable) throws Exception {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, gpuIndex);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, "PP-DocLayoutV3", gpuIndex, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, "pix2text-mfr-1.5", gpuIndex, env);
             TableModel tableModel = createTableModel(env, gpuIndex, safeTable);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            logMemory("regions-after-model-load", gpuIndex, matManager);
            Mat bgrImage = matManager.imread(imagePath);
            Mat rgbImage = matManager.newMat();
            Imgproc.cvtColor(bgrImage, rgbImage, Imgproc.COLOR_BGR2RGB);
            logMemory("regions-after-image-read", gpuIndex, matManager);

            String oriLabel = null;
            float oriScore = 0f;
            PreProcessResult oriInput = null;
            try {
                oriInput = docOriModel.processRgb(matManager, rgbImage, ndManager);
                logMemory("regions-after-doc-ori-preprocess", gpuIndex, matManager);
                ClassificationResult oriResult = docOriModel.batchPredict(
                        List.of(oriInput), 1, matManager, ndManager, Map.of()).getFirst();
                oriLabel = oriResult.label();
                oriScore = oriResult.score();
                System.out.printf(Locale.ROOT, "DOC_ORI label=%s score=%.6f%n", oriLabel, oriScore);
                logMemory("regions-after-doc-ori-predict", gpuIndex, matManager);
            } finally {
                releasePreProcessResult(matManager, oriInput);
                matManager.release(rgbImage);
            }

            Mat srcImage = bgrImage;
            if (oriLabel != null && oriScore > 0.3f) {
                double angle = Double.parseDouble(oriLabel);
                if (angle >= 1e-7) {
                    srcImage = ImageUtil.rotateImage(matManager, bgrImage, angle);
                    matManager.release(bgrImage);
                }
            }
            logMemory("regions-after-orientation-apply", gpuIndex, matManager);

            List<ObjectDetectionResult> regions;
            ProcessedMat layoutInput = null;
            Mat srcRgb = null;
            try {
                srcRgb = matManager.newMat();
                Imgproc.cvtColor(srcImage, srcRgb, Imgproc.COLOR_BGR2RGB);
                layoutInput = layoutModel.processRgb(matManager, srcRgb, ndManager);
                logMemory("regions-after-layout-preprocess", gpuIndex, matManager);
                regions = layoutModel.batchPredict(List.of(layoutInput), 1, matManager, ndManager, Map.of()).getFirst();
                logMemory("regions-after-layout-predict", gpuIndex, matManager);
            } finally {
                if (layoutInput != null) {
                    layoutInput.release(matManager);
                }
                matManager.release(srcRgb);
            }

            int formulaCount = 0;
            int tableCount = 0;
            for (int i = 0; i < regions.size(); i++) {
                ObjectDetectionResult region = regions.get(i);
                String type = classifyLabel(region.label());
                if ("formula".equals(type)) {
                    formulaCount++;
                } else if ("table".equals(type)) {
                    tableCount++;
                }
                System.out.printf(Locale.ROOT,
                        "REGION %02d type=%s label=%s score=%.6f box=%s%n",
                        i + 1, type, region.label(), region.score(), formatBox(region.coordinate()));
            }
            System.out.printf(Locale.ROOT, "REGION_SUMMARY total=%d formulas=%d tables=%d%n",
                    regions.size(), formulaCount, tableCount);

            for (int i = 0; i < regions.size(); i++) {
                ObjectDetectionResult region = regions.get(i);
                String type = classifyLabel(region.label());
                if ("formula".equals(type)) {
                    runFormulaRegion(i + 1, region, srcImage, formulaModel, matManager, ndManager, gpuIndex);
                } else if ("table".equals(type)) {
                    runTableRegion(i + 1, region, srcImage, tableModel, matManager, ndManager, gpuIndex);
                }
            }

            matManager.release(srcImage);
            logMemory("regions-after-cleanup", gpuIndex, matManager);
        }
        logMemory("regions-after-model-close", gpuIndex, null);
    }

    private static TableModel createTableModel(OrtEnvironment env, int gpuIndex, boolean safeTable) {
        if (!safeTable) {
            return new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", gpuIndex, env);
        }
        return new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", gpuIndex, env,
                Map.of(
                        "unirec.decoderGpuIndex", SAFE_TABLE_DECODER_GPU_INDEX,
                        "unirec.maxTokens", SAFE_TABLE_MAX_TOKENS
                ));
    }

    private static void runFormulaRegion(int regionIndex,
                                         ObjectDetectionResult region,
                                         Mat srcImage,
                                         FormulaRecognitionModel formulaModel,
                                         MatManager matManager,
                                         NDManager ndManager,
                                         int gpuIndex) {
        PreProcessResult input = null;
        Mat croppedRgb = null;
        long t0 = System.nanoTime();
        try {
            Mat croppedBgr = cropRegion(matManager, srcImage, region.coordinate());
            croppedRgb = matManager.newMat();
            Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
            matManager.release(croppedBgr);
            System.out.printf(Locale.ROOT, "FORMULA %02d crop=%dx%d%n",
                    regionIndex, croppedRgb.width(), croppedRgb.height());
            logMemory("formula-" + regionIndex + "-after-crop", gpuIndex, matManager);

            input = formulaModel.processRgb(matManager, croppedRgb, ndManager);
            croppedRgb = null;
            logMemory("formula-" + regionIndex + "-after-preprocess", gpuIndex, matManager);

            List<TextResult> results = formulaModel.batchPredict(List.of(input), 1, matManager, ndManager, Map.of());
            long ms = (System.nanoTime() - t0) / 1_000_000;
            TextResult result = results.isEmpty() ? null : results.getFirst();
            System.out.printf(Locale.ROOT, "FORMULA %02d ok ms=%d tokens=%d text=%s%n",
                    regionIndex, ms, result == null ? -1 : result.tokens().length,
                    result == null ? "" : abbreviate(result.text()));
            logMemory("formula-" + regionIndex + "-after-predict", gpuIndex, matManager);
        } catch (Throwable t) {
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf(Locale.ROOT, "FORMULA %02d failed ms=%d error=%s%n", regionIndex, ms, t);
            t.printStackTrace(System.out);
            logMemory("formula-" + regionIndex + "-after-failure", gpuIndex, matManager);
        } finally {
            releasePreProcessResult(matManager, input);
            matManager.release(croppedRgb);
            logMemory("formula-" + regionIndex + "-after-cleanup", gpuIndex, matManager);
        }
    }

    private static void runTableRegion(int regionIndex,
                                       ObjectDetectionResult region,
                                       Mat srcImage,
                                       TableModel tableModel,
                                       MatManager matManager,
                                       NDManager ndManager,
                                       int gpuIndex) {
        PreProcessResult input = null;
        Mat croppedRgb = null;
        long t0 = System.nanoTime();
        try {
            Mat croppedBgr = cropRegion(matManager, srcImage, region.coordinate());
            croppedRgb = matManager.newMat();
            Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
            matManager.release(croppedBgr);
            System.out.printf(Locale.ROOT, "TABLE %02d crop=%dx%d%n",
                    regionIndex, croppedRgb.width(), croppedRgb.height());
            logMemory("table-" + regionIndex + "-after-crop", gpuIndex, matManager);

            input = tableModel.processRgb(matManager, croppedRgb, ndManager);
            croppedRgb = null;
            logMemory("table-" + regionIndex + "-after-preprocess", gpuIndex, matManager);

            List<TableResult> results = tableModel.batchPredict(List.of(input), 1, matManager, ndManager, Map.of());
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf(Locale.ROOT, "TABLE %02d ok ms=%d results=%d%n",
                    regionIndex, ms, results.size());
            logMemory("table-" + regionIndex + "-after-predict", gpuIndex, matManager);
        } catch (Throwable t) {
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf(Locale.ROOT, "TABLE %02d failed ms=%d error=%s%n", regionIndex, ms, t);
            t.printStackTrace(System.out);
            logMemory("table-" + regionIndex + "-after-failure", gpuIndex, matManager);
        } finally {
            releasePreProcessResult(matManager, input);
            matManager.release(croppedRgb);
            logMemory("table-" + regionIndex + "-after-cleanup", gpuIndex, matManager);
        }
    }

    private static String renderPage(String pdfPath, int pageNumber, int dpi) throws Exception {
        try (PDDocument document = Loader.loadPDF(new File(pdfPath));
             MatManager matManager = new MatManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage bufferedImage = renderer.renderImageWithDPI(pageNumber - 1, dpi, ImageType.RGB);
            Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
            Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);

            String imagePath = System.getProperty("java.io.tmpdir") + File.separator
                    + "flux_debug_page_" + pageNumber + "_" + System.nanoTime() + ".png";
            org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, rgbMat);
            matManager.release(mat);
            matManager.release(rgbMat);
            return imagePath;
        }
    }

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
        return matManager.newMat(srcImage, new Rect(x1, y1, x2 - x1, y2 - y1));
    }

    private static String classifyLabel(String label) {
        if (FORMULA_LABELS.contains(label)) {
            return "formula";
        }
        if (TABLE_LABELS.contains(label)) {
            return "table";
        }
        return "text";
    }

    private static void releasePreProcessResult(MatManager matManager, PreProcessResult ppr) {
        if (ppr == null) {
            return;
        }
        matManager.release(ppr.mat());
        IOUtil.close(ppr.ndArray());
    }

    private static void logMemory(String label, int gpuIndex, MatManager matManager) {
        long rss = getRss(ProcessHandle.current().pid());
        long heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
        long nonHeap = ManagementFactory.getMemoryMXBean().getNonHeapMemoryUsage().getUsed();
        long gpu = getGpuMemoryMiB(gpuIndex);
        String tracked = matManager == null ? "" : String.format(Locale.ROOT,
                " mats=%d closeables=%d", matManager.trackedMatCount(), matManager.trackedCloseableCount());
        System.out.printf(Locale.ROOT,
                "MEM %-36s rss=%.1fMB heap=%.1fMB nonheap=%.1fMB gpu=%dMiB%s%n",
                label,
                rss / 1024.0 / 1024.0,
                heap / 1024.0 / 1024.0,
                nonHeap / 1024.0 / 1024.0,
                gpu,
                tracked);
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
            // fall back below
        }
        var bean = ManagementFactory.getMemoryMXBean();
        return bean.getHeapMemoryUsage().getUsed() + bean.getNonHeapMemoryUsage().getUsed();
    }

    private static long getGpuMemoryMiB(int gpuIndex) {
        if (gpuIndex < 0) {
            return -1;
        }
        try {
            Process p = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    String.valueOf(gpuIndex)).start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = br.readLine();
                if (line != null && !line.isBlank()) {
                    return Long.parseLong(line.trim());
                }
            }
            p.waitFor();
        } catch (Exception ignored) {
            return -1;
        }
        return -1;
    }

    private static String formatBox(float[] coordinate) {
        if (coordinate == null) {
            return "null";
        }
        return String.format(Locale.ROOT, "[%.1f, %.1f, %.1f, %.1f]",
                coordinate[0], coordinate[1], coordinate[2], coordinate[3]);
    }

    private static String abbreviate(String text) {
        if (text == null) {
            return "";
        }
        String normalized = text.replace("\r", "\\r").replace("\n", "\\n");
        return normalized.length() <= 120 ? normalized : normalized.substring(0, 120) + "...";
    }
}
