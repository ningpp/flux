package io.github.flux.paddle;

import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import io.github.flux.pipeline.OCRPipeline;
import io.github.flux.pipeline.OCRPipelineResult;
import io.github.flux.model.DocOrientationClassifyModel;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.model.TextRecognitionModel;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * OCR Pipeline test with text line orientation classification.
 * Outputs all intermediate results for comparison with PaddleX Python pipeline.
 */
public class OCRPipelineTextLineOriTest {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String IMAGE_PATH = "D:\\code\\flux\\scripts\\pp-ocr-pipeline\\ocr-pipeline-2026-06-12-211832.png";
    private static final String OUTPUT_DIR = "D:\\code\\flux\\scripts\\pp-ocr-pipeline";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static void main(String[] args) throws Exception {
        // Load OpenCV native library
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);

        File imageFile = new File(IMAGE_PATH);
        if (!imageFile.exists()) {
            System.err.println("Image not found: " + IMAGE_PATH);
            return;
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            // Test with CPU (gpuIndex = -1)
            System.out.println("\n========== CPU Test ==========");
            runTest(env, -1, "cpu");

            // Test with GPU (gpuIndex = 0)
            System.out.println("\n========== GPU Test ==========");
            runTest(env, 0, "gpu");
        }
    }

    private static void runTest(OrtEnvironment env, int gpuIndex, String deviceName) throws Exception {
        try (TextDetectionModel detModel = new TextDetectionModel(MODEL_ROOT_DIR, "PP-OCRv6_medium_det", env, gpuIndex);
             TextRecognitionModel recModel = new TextRecognitionModel(MODEL_ROOT_DIR, "PP-OCRv6_medium_rec", env, gpuIndex);
             DocOrientationClassifyModel docOriModel = new DocOrientationClassifyModel(MODEL_ROOT_DIR, "PP-LCNet_x1_0_doc_ori", env, gpuIndex);
             TextLineOrientationModel textLineOriModel = new TextLineOrientationModel(MODEL_ROOT_DIR, "PP-LCNet_x1_0_textline_ori", env, gpuIndex)) {

            OCRPipeline pipeline = new OCRPipeline(detModel, recModel, docOriModel, textLineOriModel, null, null, null);

            Map<String, Object> params = new HashMap<>();
            params.put("recognitionBatchSize", 1);

            List<List<OCRPipelineResult>> batchResults = pipeline.predict(List.of(IMAGE_PATH), params);
            List<OCRPipelineResult> results = batchResults.get(0);

            // Build detailed output JSON
            Map<String, Object> output = new HashMap<>();
            output.put("device", deviceName);
            output.put("image", IMAGE_PATH);

            // Doc orientation (from first result, all share same doc orientation)
            String docOriLabel = results.isEmpty() ? null : results.get(0).docOrientationLabel();
            float docOriScore = results.isEmpty() ? 0f : results.get(0).docOrientationScore();
            Map<String, Object> docOri = new HashMap<>();
            docOri.put("label", docOriLabel);
            docOri.put("score", docOriScore);
            output.put("doc_orientation", docOri);
            System.out.printf("[Doc Orientation] label=%s, score=%.4f%n", docOriLabel, docOriScore);

            // Detection
            output.put("detection_count", results.size());
            System.out.printf("[Text Detection] Found %d text regions%n", results.size());

            // Per-line results
            List<Map<String, Object>> recList = new ArrayList<>();
            for (int i = 0; i < results.size(); i++) {
                OCRPipelineResult result = results.get(i);
                String text = result.recResults().isEmpty() ? "" :
                        result.recResults().stream()
                                .map(r -> r.text())
                                .reduce("", (a, b) -> a + b);
                double confidence = result.recResults().isEmpty() ? 0.0 :
                        result.recResults().stream()
                                .mapToDouble(r -> r.scores() != null && r.scores().length > 0 ? r.scores()[0] : 0.0)
                                .average().orElse(0.0);

                Map<String, Object> recItem = new HashMap<>();
                recItem.put("index", i);
                recItem.put("text", text);
                recItem.put("confidence", Math.round(confidence * 10000.0) / 10000.0);
                recItem.put("textline_orientation", result.textLineOrientationLabel());
                recItem.put("textline_orientation_score", result.textLineOrientationScore());
                recItem.put("polygon", result.detPolys());
                recList.add(recItem);

                System.out.printf("  Line %d: text=\"%s\", conf=%.4f, ori=%s, oriScore=%.4f%n",
                        i, text, confidence,
                        result.textLineOrientationLabel(),
                        result.textLineOrientationScore());
            }
            output.put("recognition", recList);

            // Save results
            String outputPath = new File(OUTPUT_DIR, "ocr_pipeline_java_" + deviceName + "_result.json").getAbsolutePath();
            try (FileWriter writer = new FileWriter(outputPath)) {
                writer.write(GSON.toJson(output));
            }
            System.out.println("Results saved to: " + outputPath);
        }
    }
}
