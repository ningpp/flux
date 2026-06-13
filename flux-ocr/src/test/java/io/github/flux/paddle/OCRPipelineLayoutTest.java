package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
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
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * OCR Pipeline test with Layout analysis, Formula recognition, Table recognition,
 * Doc orientation classification, and Text line orientation classification.
 */
public class OCRPipelineLayoutTest {

    private static final String LAYOUT_MODEL_DIR = "D:\\models\\layout";
    private static final String FORMULA_MODEL_DIR = "D:\\models\\formula";
    private static final String OCR_MODEL_DIR = "D:\\models";
    private static final String OUTPUT_DIR = "D:\\code\\flux\\scripts\\pp-ocr-pipeline";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static void main(String[] args) throws Exception {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        String[] testImages = {
                "D:\\code\\flux\\scripts\\pp-ocr-pipeline\\layout-p32.png",
                "D:\\code\\flux\\scripts\\pp-ocr-pipeline\\layout-p31.png"
        };

        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            // Test 1: PP-DocLayoutV3 + PP-FormulaNet-L + unirec-0.1b (table) + doc ori + textline ori
            System.out.println("\n========== Test: PP-DocLayoutV3 + PP-FormulaNet-L + unirec-0.1b ==========");
            runTest(env, 0, "ppdoclayoutv3_gpu",
                    "PP-DocLayoutV3", "PP-FormulaNet-L", "unirec-0.1b",
                    "PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_textline_ori",
                    testImages);

            // Test 2: docling-layout-heron + unirec-0.1b (formula + table) + doc ori + textline ori
            System.out.println("\n========== Test: docling-layout-heron + unirec-0.1b ==========");
            runTest(env, 0, "docling_gpu",
                    "docling-layout-heron", "unirec-0.1b", "unirec-0.1b",
                    "PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_textline_ori",
                    testImages);

            // Test 3: Rotation tests for both layout-p31.png and layout-p32.png (90, 180, 270 degrees)
            System.out.println("\n========== Test: Rotation tests ==========");
            runRotationTest(env, 0, "ppdoclayoutv3_gpu",
                    "PP-DocLayoutV3", "PP-FormulaNet-L", "unirec-0.1b",
                    "PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_textline_ori",
                    testImages);
        }
    }

    private static void runTest(OrtEnvironment env, int gpuIndex, String deviceName,
                                String layoutModelName, String formulaModelName, String tableModelName,
                                String docOriModelName, String textLineOriModelName,
                                String[] testImages) throws Exception {
        try (TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, docOriModelName, env, gpuIndex);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(OCR_MODEL_DIR, textLineOriModelName, env, gpuIndex);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, layoutModelName, gpuIndex, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, formulaModelName, gpuIndex, env);
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, tableModelName, gpuIndex, env)) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel,
                    docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);

            for (String imagePath : testImages) {
                processImage(pipeline, List.of(imagePath), deviceName, layoutModelName, formulaModelName, tableModelName);
            }
        }
    }

    private static void runRotationTest(OrtEnvironment env, int gpuIndex, String deviceName,
                                         String layoutModelName, String formulaModelName, String tableModelName,
                                         String docOriModelName, String textLineOriModelName,
                                         String[] testImages) throws Exception {
        try (TextDetectionModel detModel = new TextDetectionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(OCR_MODEL_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(OCR_MODEL_DIR, docOriModelName, env, gpuIndex);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(OCR_MODEL_DIR, textLineOriModelName, env, gpuIndex);
             LayoutModel layoutModel = new LayoutModel(LAYOUT_MODEL_DIR, layoutModelName, gpuIndex, env);
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(FORMULA_MODEL_DIR, formulaModelName, gpuIndex, env);
             TableModel tableModel = new TableModel(FORMULA_MODEL_DIR, tableModelName, gpuIndex, env);
             NDManager ndManager = NDManager.newBaseManager();
             MatManager matManager = new MatManager()) {

            OCRPipeline pipeline = new OCRPipeline(
                    detModel, recModel,
                    docOriModel, textLineOriModel,
                    layoutModel, formulaModel, tableModel);

            for (String imagePath : testImages) {
                System.out.println("\n====== Rotation test for: " + imagePath + " ======");

                // 1. Get baseline result (original image)
                Map<String, Object> params = new HashMap<>();
                params.put("recognitionBatchSize", 1);
                List<List<OCRPipelineResult>> baselineResults = pipeline.predict(List.of(imagePath), params);
                List<OCRPipelineResult> baseline = baselineResults.get(0);
                String baselineText = extractAllText(baseline);
                System.out.println("  Baseline text length: " + baselineText.length());

                // 2. Generate rotated images using ImageUtil.rotateImage
                double[] angles = {90.0, 180.0, 270.0};
                List<String> rotatedPaths = new ArrayList<>();
                Mat originalBgr = matManager.imread(imagePath, Imgcodecs.IMREAD_COLOR_BGR);

                for (double angle : angles) {
                    String rotatedPath = imagePath.replace(".png", "_rot" + (int) angle + ".png");
                    Mat rotated = ImageUtil.rotateImage(matManager, originalBgr, angle);
                    Imgcodecs.imwrite(rotatedPath, rotated);
                    rotatedPaths.add(rotatedPath);
                    System.out.println("  Generated rotated image: " + rotatedPath);
                }

                // 3. Batch predict all rotated images
                List<List<OCRPipelineResult>> rotatedResults = pipeline.predict(rotatedPaths, params);

                // 4. Compare each rotated result with baseline
                for (int i = 0; i < angles.length; i++) {
                    double angle = angles[i];
                    List<OCRPipelineResult> rotatedResult = rotatedResults.get(i);
                    String rotatedText = extractAllText(rotatedResult);

                    // Check doc orientation was correctly detected
                    String docOriLabel = rotatedResult.isEmpty() ? null : rotatedResult.get(0).docOrientationLabel();
                    float docOriScore = rotatedResult.isEmpty() ? 0f : rotatedResult.get(0).docOrientationScore();

                    System.out.printf("  Rotation %.0f: docOri=%s(%.4f), textLen=%d, baselineTextLen=%d%n",
                            angle, docOriLabel, docOriScore, rotatedText.length(), baselineText.length());

                    // Verify doc orientation classification detected the correct correction angle.
                    // The doc orientation model outputs the "correction angle" — the angle to rotate
                    // the image back to its upright orientation — not the angle by which the image
                    // was originally rotated. For example, if the image was rotated 90° clockwise,
                    // the model outputs "270" because rotating 270° clockwise (or 90° counterclockwise)
                    // restores the image. So the expected correction angle = (360 - rotationAngle) % 360.
                    int expectedCorrectionAngle = (360 - (int) angle) % 360;
                    String expectedOri = String.valueOf(expectedCorrectionAngle);
                    if (!expectedOri.equals(docOriLabel)) {
                        System.out.printf("  WARNING: Expected doc orientation correction '%s' but got '%s' (score=%.4f)%n",
                                expectedOri, docOriLabel, docOriScore);
                    }

                    // Compare text content consistency
                    double similarity = computeTextSimilarity(baselineText, rotatedText);
                    System.out.printf("  Rotation %.0f: text similarity with baseline = %.2f%%%n", angle, similarity * 100);
                    if (similarity < 0.95) {
                        System.out.printf("  WARNING: Text similarity %.2f%% is below 95%% after rotation %.0f correction!%n", similarity * 100, angle);
                        System.out.println("    Baseline: " + baselineText);
                        System.out.println("    Rotated : " + rotatedText);
                    }

                    // Save rotated result
                    saveResult(rotatedResult, rotatedPaths.get(i), deviceName + "_rot" + (int) angle,
                            layoutModelName, formulaModelName, tableModelName);
                }

                // 5. Clean up rotated images
                for (String rotatedPath : rotatedPaths) {
                    new File(rotatedPath).delete();
                }
            }
        }
    }

    /**
     * Extract all text content from OCR pipeline results.
     */
    private static String extractAllText(List<OCRPipelineResult> results) {
        StringBuilder sb = new StringBuilder();
        for (OCRPipelineResult result : results) {
            if (result.layoutRegions() != null) {
                for (LayoutRegionResult region : result.layoutRegions()) {
                    sb.append(region.getText());
                }
            }
            if (result.recResults() != null) {
                for (var rec : result.recResults()) {
                    sb.append(rec.text());
                }
            }
        }
        return sb.toString();
    }

    /**
     * Compute text similarity between two strings using character-level comparison.
     * Returns a value between 0.0 (completely different) and 1.0 (identical).
     */
    private static double computeTextSimilarity(String a, String b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        if (a.isEmpty() || b.isEmpty()) return 0.0;

        // Normalize whitespace for comparison
        String na = a.replaceAll("\\s+", " ").trim();
        String nb = b.replaceAll("\\s+", " ").trim();

        if (na.equals(nb)) return 1.0;

        // Use Levenshtein distance ratio
        int maxLen = Math.max(na.length(), nb.length());
        int dist = levenshteinDistance(na, nb);
        return 1.0 - (double) dist / maxLen;
    }

    private static int levenshteinDistance(String a, String b) {
        int m = a.length();
        int n = b.length();
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];

        for (int j = 0; j <= n; j++) prev[j] = j;

        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                curr[j] = Math.min(Math.min(prev[j] + 1, curr[j - 1] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[n];
    }

    private static void processImage(OCRPipeline pipeline, List<String> imagePaths,
                                     String deviceName, String layoutModelName,
                                     String formulaModelName, String tableModelName) throws Exception {
        for (String imagePath : imagePaths) {
            File imageFile = new File(imagePath);
            if (!imageFile.exists()) {
                System.err.println("Image not found: " + imagePath);
                return;
            }
        }

        Map<String, Object> params = new HashMap<>();
        params.put("recognitionBatchSize", 1);

        System.out.println("\n--- Processing: " + imagePaths + " ---");
        List<List<OCRPipelineResult>> batchResults = pipeline.predict(imagePaths, params);

        for (int imgIdx = 0; imgIdx < imagePaths.size(); imgIdx++) {
            String imagePath = imagePaths.get(imgIdx);
            List<OCRPipelineResult> results = batchResults.get(imgIdx);
            saveResult(results, imagePath, deviceName, layoutModelName, formulaModelName, tableModelName);
        }
    }

    private static void saveResult(List<OCRPipelineResult> results, String imagePath,
                                    String deviceName, String layoutModelName,
                                    String formulaModelName, String tableModelName) throws Exception {
        Map<String, Object> output = new HashMap<>();
        output.put("device", deviceName);
        output.put("layoutModel", layoutModelName);
        output.put("formulaModel", formulaModelName);
        output.put("tableModel", tableModelName);
        output.put("image", imagePath);

        List<Map<String, Object>> regionList = new ArrayList<>();
        for (OCRPipelineResult result : results) {
            if (result.layoutRegions() != null) {
                for (LayoutRegionResult region : result.layoutRegions()) {
                    Map<String, Object> regionMap = new HashMap<>();
                    regionMap.put("label", region.layoutRegion().label());
                    regionMap.put("score", region.layoutRegion().score());
                    regionMap.put("regionType", region.regionType());
                    regionMap.put("coordinate", region.layoutRegion().coordinate());

                    String content = region.getText();
                    if (content.length() > 500) {
                        content = content.substring(0, 500) + "...";
                    }
                    regionMap.put("content", content);

                    if ("text".equals(region.regionType()) && region.textResults() != null) {
                        regionMap.put("textLineCount", region.textResults().size());
                    }

                    regionList.add(regionMap);

                    System.out.printf("  [%s] label=%s, score=%.4f, content=%s%n",
                            region.regionType(),
                            region.layoutRegion().label(),
                            region.layoutRegion().score(),
                            content.length() > 100 ? content.substring(0, 100) + "..." : content);
                }
            }

            // Print doc orientation info
            if (result.docOrientationLabel() != null) {
                System.out.printf("  [DocOrientation] label=%s, score=%.4f%n",
                        result.docOrientationLabel(), result.docOrientationScore());
            }
        }
        output.put("regions", regionList);
        output.put("regionCount", regionList.size());

        // Doc orientation info
        String docOriLabel = results.isEmpty() ? null : results.get(0).docOrientationLabel();
        float docOriScore = results.isEmpty() ? 0f : results.get(0).docOrientationScore();
        if (docOriLabel != null) {
            Map<String, Object> docOri = new HashMap<>();
            docOri.put("label", docOriLabel);
            docOri.put("score", docOriScore);
            output.put("docOrientation", docOri);
        }

        String baseName = new File(imagePath).getName().replace(".png", "");
        String outputPath = new File(OUTPUT_DIR,
                "ocr_pipeline_layout_" + deviceName + "_" + baseName + "_result.json").getAbsolutePath();
        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.write(GSON.toJson(output));
        }
        System.out.println("Results saved to: " + outputPath);
    }
}
