package io.github.flux.llava;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.llava.LlavaOneVisionImageProcessor.ImageProcessResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Batch inference test for LLaVA-OneVision-Qwen2-0.5B.
 * Tests batch inference and compares with sequential results.
 */
public class LlavaOneVisionBatchTest {

    public static void main(String[] args) throws Exception {
        String modelRootDir = "D:\\models";
        String modelName = "llava-onevision-qwen2-0.5b-ov-hf";
        int gpuIndex = -1;  // CPU
        int batchSize = 2;

        List<String> imagePaths = List.of(
                "D:\\tmp\\formula_2025-8-2_17-28-16.jpg",
                "D:\\tmp\\table-2026-01-01-202211.png"
        );

        System.out.println("=".repeat(60));
        System.out.println("LLaVA-OneVision-Qwen2-0.5B — Batch Inference Test");
        System.out.println("=".repeat(60));
        System.out.println("Batch size: " + batchSize);
        System.out.println("Images: " + imagePaths.size());

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            LlavaOneVisionQwenModel model = new LlavaOneVisionQwenModel(modelRootDir, modelName, gpuIndex, env);
            System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + "ms");

            // Load all images
            List<Mat> rgbMats = new ArrayList<>();
            List<ImageProcessResult> processResults = new ArrayList<>();

            for (String path : imagePaths) {
                Mat rgbMat = ImageUtil.readToRgb(matManager, path);
                System.out.println("Loaded: " + path + " (" + rgbMat.cols() + "x" + rgbMat.rows() + ")");
                rgbMats.add(rgbMat);
                processResults.add(model.processRgb(matManager, rgbMat, ndManager));
            }

            // Batch inference
            System.out.println("\n--- BATCH INFERENCE ---");
            long t1 = System.currentTimeMillis();
            List<TextResult> batchResults = model.doBatchPredict(processResults, matManager, ndManager, null);
            long batchTime = System.currentTimeMillis() - t1;
            System.out.println("Batch inference completed in " + batchTime + "ms");

            for (int i = 0; i < batchResults.size(); i++) {
                TextResult result = batchResults.get(i);
                System.out.println("\nImage " + (i + 1) + ":");
                System.out.println("Tokens: " + result.tokens().length);
                System.out.println("Text: " + result.text());
            }

            // Sequential inference for comparison
            System.out.println("\n--- SEQUENTIAL INFERENCE (for comparison) ---");
            List<TextResult> sequentialResults = new ArrayList<>();
            long t2 = System.currentTimeMillis();

            for (int i = 0; i < processResults.size(); i++) {
                List<TextResult> singleResult = model.doBatchPredict(
                        List.of(processResults.get(i)), matManager, ndManager, null);
                sequentialResults.add(singleResult.get(0));
                System.out.println("Image " + (i + 1) + ": " + singleResult.get(0).text());
            }

            long sequentialTime = System.currentTimeMillis() - t2;
            System.out.println("\nSequential inference completed in " + sequentialTime + "ms");

            // Compare results
            System.out.println("\n--- COMPARISON ---");
            System.out.println("Batch time: " + batchTime + "ms (" + (batchTime / batchSize) + "ms per image)");
            System.out.println("Sequential time: " + sequentialTime + "ms (" + (sequentialTime / batchSize) + "ms per image)");

            boolean allMatch = true;
            for (int i = 0; i < batchResults.size(); i++) {
                long[] batchTokens = batchResults.get(i).tokens();
                long[] seqTokens = sequentialResults.get(i).tokens();

                if (batchTokens.length != seqTokens.length) {
                    System.out.println("Image " + (i + 1) + ": Token count mismatch (batch=" + batchTokens.length + ", sequential=" + seqTokens.length + ")");
                    allMatch = false;
                } else {
                    boolean match = true;
                    for (int j = 0; j < batchTokens.length; j++) {
                        if (batchTokens[j] != seqTokens[j]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        System.out.println("Image " + (i + 1) + ": Tokens MATCH ✓");
                    } else {
                        System.out.println("Image " + (i + 1) + ": Tokens MISMATCH ✗");
                        allMatch = false;
                    }
                }
            }

            if (allMatch) {
                System.out.println("\n✓ All batch results match sequential results!");
            } else {
                System.out.println("\n✗ Some batch results differ from sequential results");
            }

            model.close();
        }
    }
}
