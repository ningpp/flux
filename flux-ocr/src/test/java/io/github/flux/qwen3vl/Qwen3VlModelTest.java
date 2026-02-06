package io.github.flux.qwen3vl;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.Arrays;
import java.util.List;

/**
 * Test for Qwen3-VL-2B-Instruct Java ONNX inference.
 * Validates output matches Python ONNX reference:
 *   Ground truth: "```latex\nn! \approx \sqrt{2\pi n} \left( \frac{n}{e} \right)^n\n```"
 *   Token IDs: [73594, 64680, 198, 77, 0, 1124, 48053, 1124, 26888, 90, 17, 59,
 *               2493, 308, 92, 1124, 2359, 7, 1124, 37018, 91362, 15170, 68, 92,
 *               1124, 1291, 29776, 77, 198, 73594, 151645]
 */
public class Qwen3VlModelTest {

    private static final long[] GROUND_TRUTH_IDS = {
            73594, 64680, 198, 77, 0, 1124, 48053, 1124, 26888, 90, 17, 59,
            2493, 308, 92, 1124, 2359, 7, 1124, 37018, 91362, 15170, 68, 92,
            1124, 1291, 29776, 77, 198, 73594, 151645
    };

    public static void main(String[] args) throws Exception {
        String modelRootDir = "D:\\models\\onnx";
        String modelName = "Qwen3-VL-2B-Instruct";
        String imagePath = "D:\\tmp\\formula_2025-8-2_17-28-16.jpg";
        int gpuIndex = -1;  // CPU

        System.out.println("=".repeat(60));
        System.out.println("Qwen3-VL-2B-Instruct  —  Java ONNX Inference (greedy)");
        System.out.println("=".repeat(60));

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            Qwen3VlModel model = new Qwen3VlModel(modelRootDir, modelName, gpuIndex, env);
            System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + "ms");

            long t1 = System.currentTimeMillis();
            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            System.out.println("Image: " + rgbMat.cols() + "x" + rgbMat.rows());

            PreProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);

            List<TextResult> results = model.doBatchPredict(
                    List.of(ppr), matManager, ndManager, null);

            long totalTime = System.currentTimeMillis() - t1;

            for (TextResult result : results) {
                long[] tokens = result.tokens();
                System.out.println("\nGenerated " + tokens.length + " tokens in " + totalTime + "ms");

                System.out.print("Token IDs: [");
                for (int i = 0; i < tokens.length; i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.print(tokens[i]);
                }
                System.out.println("]");

                System.out.println("\n--- GENERATED TEXT ---");
                System.out.println(result.text());
                System.out.println("--- END ---");

                // Validate against ground truth
                System.out.println("\n--- VALIDATION ---");
                int matchCount = 0;
                int compareLen = Math.min(tokens.length, GROUND_TRUTH_IDS.length);
                for (int i = 0; i < compareLen; i++) {
                    if (tokens[i] == GROUND_TRUTH_IDS[i]) matchCount++;
                }
                System.out.println("  Match: " + matchCount + "/" + GROUND_TRUTH_IDS.length + " tokens");
                if (matchCount == GROUND_TRUTH_IDS.length && tokens.length == GROUND_TRUTH_IDS.length) {
                    System.out.println("  PERFECT MATCH!");
                } else {
                    System.out.println("  MISMATCH!");
                    System.out.println("  Expected: " + Arrays.toString(Arrays.copyOf(GROUND_TRUTH_IDS, 15)) + "...");
                    System.out.println("  Got:      " + Arrays.toString(Arrays.copyOf(tokens, Math.min(15, tokens.length))) + "...");
                }
            }

            model.close();
        }
    }
}
