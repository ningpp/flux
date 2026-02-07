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
            System.out.println("Image: " + rgbMat.cols() + "x" + rgbMat.rows());

            ImageProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);

            List<TextResult> results = model.doBatchPredict(
                    List.of(ppr), matManager, ndManager, null);

            long totalTime = System.currentTimeMillis() - t1;

            for (TextResult result : results) {
                long[] tokens = result.tokens();
                System.out.println("\nGenerated " + tokens.length + " tokens in " + totalTime + "ms");

                System.out.print("Token IDs: [");
                for (int i = 0; i < Math.min(20, tokens.length); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.print(tokens[i]);
                }
                if (tokens.length > 20) {
                    System.out.print(", ... (" + (tokens.length - 20) + " more)");
                }
                System.out.println("]");

                System.out.println("\n--- GENERATED TEXT ---");
                System.out.println(result.text());
                System.out.println("--- END ---");
            }

            model.close();
        }
    }
}
