package io.github.flux.llava;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.llava.LlavaOneVisionImageProcessor.ImageProcessResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;

/**
 * Test for LLaVA-OneVision-Qwen2-0.5B Java ONNX inference.
 * Tests single image inference and measures performance.
 * 
 * This test outputs detailed validation information that can be compared
 * with the Python reference implementation to ensure correctness.
 * 
 * Validation workflow:
 * 1. Run this test and save output: java LlavaOneVisionQwenModelTest > java_output.log
 * 2. Run Python script: python llava_onevision_onnx_infer.py > python_output.log
 * 3. Compare outputs: python validate_llava_outputs.py java_output.log python_output.log
 * 
 * Or use the automated script: scripts\validate_llava.bat
 * 
 * @see scripts/llava_onevision_onnx_infer.py - Python reference implementation
 * @see scripts/validate_llava_outputs.py - Validation comparison script
 * @see scripts/VALIDATION_LLAVA.md - Detailed validation guide
 */
public class LlavaOneVisionQwenModelTest {

    public static void main(String[] args) throws Exception {
        String modelRootDir = "D:\\models";
        String modelName = "llava-onevision-qwen2-0.5b-ov-hf";
        String imagePath = "D:\\tmp\\formula_2025-8-2_17-28-16.jpg";
        imagePath = "D:\\tmp\\img-2026-02-07-120114.png";
        int gpuIndex = -1;  // CPU

        System.out.println("=".repeat(60));
        System.out.println("LLaVA-OneVision-Qwen2-0.5B — Java ONNX Inference");
        System.out.println("=".repeat(60));

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            LlavaOneVisionQwenModel model = new LlavaOneVisionQwenModel(modelRootDir, modelName, gpuIndex, env);
            System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + "ms");

            long t1 = System.currentTimeMillis();
            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            System.out.println("\nImage: " + rgbMat.cols() + "x" + rgbMat.rows());

            ImageProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);
            
            // Print preprocessing details for validation
            System.out.println("\n--- Preprocessing Details ---");
            float[][] pixelValues = ppr.pixelValues();
            System.out.println("Pixel values shape: [" + pixelValues.length + ", " + pixelValues[0].length + "]");
            float minVal = Float.MAX_VALUE, maxVal = Float.MIN_VALUE;
            for (float[] row : pixelValues) {
                for (float val : row) {
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
            }
            System.out.printf("Pixel values range: [%.3f, %.3f]\n", minVal, maxVal);
            
            // Print first few pixel values for debugging
            System.out.print("First 10 pixel values (channel 0): [");
            for (int i = 0; i < Math.min(10, pixelValues[0].length); i++) {
                if (i > 0) System.out.print(", ");
                System.out.printf("%.3f", pixelValues[0][i]);
            }
            System.out.println("]");

            // Configure parameters
            // Use default prompt "OCR" or customize with: params.put("prompt", "Describe this image in detail.");
            java.util.Map<String, Object> params = new java.util.HashMap<>();
            params.put("debug", true);
            params.put("prompt", "Convert this image to Latex.");
            params.put("prompt", "Describe this image");

            List<TextResult> results = model.doBatchPredict(
                    List.of(ppr), matManager, ndManager, params);

            long totalTime = System.currentTimeMillis() - t1;

            for (TextResult result : results) {
                long[] tokens = result.tokens();
                System.out.println("\n--- Generation Results ---");
                System.out.println("Generated " + tokens.length + " tokens in " + totalTime + "ms");

                System.out.print("Token IDs: [");
                for (int i = 0; i < Math.min(20, tokens.length); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.print(tokens[i]);
                }
                if (tokens.length > 20) {
                    System.out.print(", ... (" + (tokens.length - 20) + " more)");
                }
                System.out.println("]");
                
                // Print all token IDs for validation
                System.out.println("\nAll token IDs:");
                for (int i = 0; i < tokens.length; i++) {
                    if (i > 0 && i % 20 == 0) System.out.println();
                    System.out.print(tokens[i] + " ");
                }

                System.out.println("\n\n--- GENERATED TEXT ---");
                System.out.println(result.text());
                System.out.println("--- END ---");
            }

            model.close();
        }
    }
}
