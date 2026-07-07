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
import io.github.flux.pipeline.OCRPipeline;
import io.github.flux.pipeline.OCRPipelineResult;
import io.github.flux.util.ImageUtil;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.opencv.core.Mat;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * OCRPipeline full-stack memory leak + performance verifier.
 * <p>
 * Uses the same concrete models as {@link OCRPipelineGPUPerfTest}:
 * PP-OCRv6_medium_det, PP-OCRv6_medium_rec, PP-LCNet_x1_0_doc_ori,
 * PP-LCNet_x1_0_textline_ori, PP-DocLayoutV3, pix2text-mfr-1.5, unirec-0.1b.
 * <p>
 * Usage:
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # optional: <pdfPath> <iterations> <gpuIndex> <dpi>
 */
public class OCRPipelineLeakPerfVerify {

    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final String DEFAULT_PDF = "E:\\flux-data\\2606.13108_zh_CN.pdf";
    private static final Pattern RSS_PATTERN = Pattern.compile("([\\d,]+)\\s*K");

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String pdfPath = args.length > 0 ? args[0] : DEFAULT_PDF;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 5;
        int gpuIndex = args.length > 2 ? Integer.parseInt(args[2]) : 0;
        int dpi = args.length > 3 ? Integer.parseInt(args[3]) : 300;
        long pid = ProcessHandle.current().pid();

        if (!new File(pdfPath).exists()) {
            throw new IllegalArgumentException("PDF file not found: " + pdfPath);
        }

        System.out.printf(Locale.ROOT,
                "PDF=%s  iterations=%d  gpuIndex=%d  dpi=%d  pid=%d%n",
                pdfPath, iterations, gpuIndex, dpi, pid);

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_doc_ori", env, gpuIndex);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(OCR_MODEL_DIR, "PP-LCNet_x1_0_textline_ori", env, gpuIndex);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, "PP-DocLayoutV3", gpuIndex, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, "pix2text-mfr-1.5", gpuIndex, env);
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, "unirec-0.1b", gpuIndex, env)) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel, docOriModel, textLineOriModel, layoutModel, formulaModel, tableModel);

            Map<String, Object> params = new HashMap<>();
            params.put("recognitionBatchSize", 4);
            params.put("formulaBatchSize", 1);

            // Warm up with first page.
            List<String> warmupImages = convertPdfToImages(pdfPath, dpi, 1, "flux_ocr_warmup");
            try {
                pipeline.predict(List.of(warmupImages.get(0)), params);
            } finally {
                deleteFiles(warmupImages);
            }
            System.gc();

            long rssBaseline = getRss(pid);
            long gpuBaseline = getGpuMemoryMiB(gpuIndex);
            long totalOcrMs = 0;
            int totalPages = 0;

            System.out.printf(Locale.ROOT, "baseline rss=%.1f MB gpu=%d MiB%n",
                    rssBaseline / 1024.0 / 1024.0, gpuBaseline);

            for (int it = 0; it < iterations; it++) {
                List<String> imagePaths = new ArrayList<>();
                try {
                    imagePaths = convertPdfToImages(pdfPath, dpi, Integer.MAX_VALUE, "flux_ocr_leak_" + it);
                    long t0 = System.nanoTime();
                    int regions = 0;
                    for (String imagePath : imagePaths) {
                        List<List<OCRPipelineResult>> results = pipeline.predict(List.of(imagePath), params);
                        regions += results.stream().mapToInt(List::size).sum();
                    }
                    long t1 = System.nanoTime();
                    long iterMs = (t1 - t0) / 1_000_000;
                    totalOcrMs += iterMs;
                    totalPages += imagePaths.size();

                    System.gc();
                    Thread.sleep(200);
                    long rss = getRss(pid);
                    long gpu = getGpuMemoryMiB(gpuIndex);
                    System.out.printf(Locale.ROOT,
                            "iter=%02d pages=%d regions=%d ocr=%.1f ms avg=%.1f ms/page rss=%.1f MB delta=%.1f MB gpu=%d MiB delta=%d MiB%n",
                            it + 1,
                            imagePaths.size(),
                            regions,
                            (double) iterMs,
                            imagePaths.isEmpty() ? 0 : (double) iterMs / imagePaths.size(),
                            rss / 1024.0 / 1024.0,
                            (rss - rssBaseline) / 1024.0 / 1024.0,
                            gpu,
                            gpu < 0 || gpuBaseline < 0 ? -1 : gpu - gpuBaseline);
                } finally {
                    deleteFiles(imagePaths);
                }
            }

            System.out.printf(Locale.ROOT, "total pages=%d avg OCR=%.1f ms/page%n",
                    totalPages, totalPages == 0 ? 0 : (double) totalOcrMs / totalPages);
        }
    }

    private static List<String> convertPdfToImages(String pdfFile, int dpi, int maxPages, String prefix) throws Exception {
        List<String> imagePaths = new ArrayList<>();
        try (PDDocument document = Loader.loadPDF(new File(pdfFile));
             MatManager matManager = new MatManager()) {
            PDFRenderer renderer = new PDFRenderer(document);
            int pageCount = Math.min(document.getNumberOfPages(), maxPages);
            for (int i = 0; i < pageCount; i++) {
                BufferedImage bufferedImage = renderer.renderImageWithDPI(i, dpi, ImageType.RGB);
                Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
                Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);
                String imagePath = System.getProperty("java.io.tmpdir") + File.separator
                        + prefix + "_page_" + (i + 1) + ".png";
                org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, rgbMat);
                imagePaths.add(imagePath);
                matManager.release(rgbMat);
            }
        }
        return imagePaths;
    }

    private static void deleteFiles(List<String> paths) {
        for (String path : paths) {
            new File(path).delete();
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
}
