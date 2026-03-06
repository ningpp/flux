package io.github.flux.lightonocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Test harness for LightOnOCR-2-1B ONNX inference.
 */
public class LightOnOcrModelTest {

    public static void main(String[] args) throws Exception {
        String modelRootDir = "D:\\models";
        String modelName = "LightOnOCR-2-1B-ONNX";
        String imagePath = "D:\\tmp\\formula-2026-01-18-152316.png";
        int gpuIndex = 0;  // CPU only

        System.out.println("=".repeat(60));
        System.out.println("LightOnOCR-2-1B  —  Java ONNX Inference (greedy)");
        System.out.println("=".repeat(60));
        int n = 31;
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             LightOnOcrModel model = new LightOnOcrModel(modelRootDir, modelName, gpuIndex, env)) {
            List<TextResult> results = null;
            LocalDateTime start = LocalDateTime.now();
            for (int j = 0; j < n; j++) {
                try (MatManager matManager = new MatManager();
                     NDManager ndManager = NDManager.newBaseManager()) {
                    // Read image and convert to RGB
                    long t1 = System.currentTimeMillis();
                    Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);

                    // Preprocess
                    PreProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);

                    // Run inference
                    results = model.doBatchPredict(
                            List.of(ppr), matManager, ndManager, null);

                    long totalTime = System.currentTimeMillis() - t1;

                    for (TextResult result : results) {
                        System.out.println("\nGenerated " + result.tokens().length + " tokens in " + totalTime + "ms");
                        System.out.print("First 10 token IDs: [");
                        long[] tokens = result.tokens();
                        for (int i = 0; i < Math.min(10, tokens.length); i++) {
                            if (i > 0) System.out.print(", ");
                            System.out.print(tokens[i]);
                        }
                        System.out.println("]");
                        System.out.println("\n--- GENERATED TEXT ---");
                        System.out.println(result.text());
                        System.out.println("--- END ---");
                    }
                }
            }
            var end = LocalDateTime.now();
            double avgCostMs = Duration.between(start, end).toNanos() / 1000_000d / (double) n;
            System.out.println("Avg Cost: " + avgCostMs+ "ms");
            System.out.println("Generated Tokens: " + results.get(0).tokens().length);
            System.out.println(results.get(0).tokens().length / (avgCostMs / 1000.0) + " Tokens/s");
            System.out.println(start + "\t\t" + end);
        }
    }
}
