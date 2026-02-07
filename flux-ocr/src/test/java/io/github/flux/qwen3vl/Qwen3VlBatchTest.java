package io.github.flux.qwen3vl;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.qwen3vl.Qwen3VlImageProcessor.ImageProcessResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.Arrays;
import java.util.List;

/**
 * Batch inference test for Qwen3-VL-2B-Instruct Java ONNX.
 *
 * Tests:
 *   1. Batch=2 with two small random images — verifies no errors
 *   2. Batch=2 vs sequential — verifies outputs match single inference
 *   3. Batch=1 with real formula image — verifies ground truth match
 */
public class Qwen3VlBatchTest {

    // Force OpenCV native library loading
    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

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
        System.out.println("Qwen3-VL  Java Batch Inference Test");
        System.out.println("=".repeat(60));

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            Qwen3VlModel model = new Qwen3VlModel(modelRootDir, modelName, gpuIndex, env);
            System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + "ms");

            // --- Test 1: Batch=2 with random images ---
            System.out.println("\n" + "=".repeat(60));
            System.out.println("TEST 1: Batch=2 with two random images");
            System.out.println("=".repeat(60));
            {
                Mat img1 = createRandomRgbMat(matManager, 100, 200);
                Mat img2 = createRandomRgbMat(matManager, 200, 100);

                ImageProcessResult ppr1 = model.processRgb(matManager, img1, ndManager);
                ImageProcessResult ppr2 = model.processRgb(matManager, img2, ndManager);

                long t1 = System.currentTimeMillis();
                List<TextResult> batchResults = model.doBatchPredict(
                        List.of(ppr1, ppr2), matManager, ndManager, null);
                long batchTime = System.currentTimeMillis() - t1;

                System.out.println("  Batch=2 completed in " + batchTime + "ms");
                for (int i = 0; i < batchResults.size(); i++) {
                    TextResult r = batchResults.get(i);
                    System.out.println("  Image " + i + ": [" + r.tokens().length + " tokens] "
                            + truncate(r.text(), 80));
                }
                System.out.println("  TEST 1: PASSED (no errors)");
            }

            // --- Test 2: Batch=2 vs sequential ---
            System.out.println("\n" + "=".repeat(60));
            System.out.println("TEST 2: Batch=2 vs sequential (same images)");
            System.out.println("=".repeat(60));
            {
                // Use deterministic "random" images (filled with constant values)
                Mat img1 = createFilledRgbMat(matManager, 100, 200, 42);
                Mat img2 = createFilledRgbMat(matManager, 200, 100, 99);

                // Batch inference
                ImageProcessResult ppr1 = model.processRgb(matManager, img1, ndManager);
                ImageProcessResult ppr2 = model.processRgb(matManager, img2, ndManager);
                List<TextResult> batchResults = model.doBatchPredict(
                        List.of(ppr1, ppr2), matManager, ndManager, null);

                // Sequential inference: image 1
                ImageProcessResult pprSeq1 = model.processRgb(matManager, img1, ndManager);
                List<TextResult> seqResult1 = model.doBatchPredict(
                        List.of(pprSeq1), matManager, ndManager, null);

                // Sequential inference: image 2
                ImageProcessResult pprSeq2 = model.processRgb(matManager, img2, ndManager);
                List<TextResult> seqResult2 = model.doBatchPredict(
                        List.of(pprSeq2), matManager, ndManager, null);

                boolean match1 = Arrays.equals(batchResults.get(0).tokens(), seqResult1.get(0).tokens());
                boolean match2 = Arrays.equals(batchResults.get(1).tokens(), seqResult2.get(0).tokens());

                System.out.println("  Image 0 batch vs seq match: " + match1);
                System.out.println("    Batch:      " + Arrays.toString(
                        Arrays.copyOf(batchResults.get(0).tokens(), Math.min(15, batchResults.get(0).tokens().length))));
                System.out.println("    Sequential: " + Arrays.toString(
                        Arrays.copyOf(seqResult1.get(0).tokens(), Math.min(15, seqResult1.get(0).tokens().length))));
                System.out.println("  Image 1 batch vs seq match: " + match2);
                System.out.println("    Batch:      " + Arrays.toString(
                        Arrays.copyOf(batchResults.get(1).tokens(), Math.min(15, batchResults.get(1).tokens().length))));
                System.out.println("    Sequential: " + Arrays.toString(
                        Arrays.copyOf(seqResult2.get(0).tokens(), Math.min(15, seqResult2.get(0).tokens().length))));

                if (match1 && match2) {
                    System.out.println("  TEST 2: PASSED (BATCH == SEQUENTIAL)");
                } else {
                    System.out.println("  TEST 2: WARNING (batch and sequential differ)");
                }
            }

            // --- Test 3: Batch=1 with real formula image (ground truth) ---
            System.out.println("\n" + "=".repeat(60));
            System.out.println("TEST 3: Batch=1 real formula image (ground truth)");
            System.out.println("=".repeat(60));
            {
                Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
                System.out.println("  Image: " + rgbMat.cols() + "x" + rgbMat.rows());

                ImageProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);
                List<TextResult> results = model.doBatchPredict(
                        List.of(ppr), matManager, ndManager, null);

                long[] tokens = results.get(0).tokens();
                int matchCount = 0;
                int compareLen = Math.min(tokens.length, GROUND_TRUTH_IDS.length);
                for (int i = 0; i < compareLen; i++) {
                    if (tokens[i] == GROUND_TRUTH_IDS[i]) matchCount++;
                }
                System.out.println("  Generated: " + tokens.length + " tokens");
                System.out.println("  Text: " + results.get(0).text());
                System.out.println("  Match: " + matchCount + "/" + GROUND_TRUTH_IDS.length);
                if (matchCount == GROUND_TRUTH_IDS.length && tokens.length == GROUND_TRUTH_IDS.length) {
                    System.out.println("  TEST 3: PASSED (PERFECT MATCH)");
                } else {
                    System.out.println("  TEST 3: FAILED");
                    System.out.println("  Expected: " + Arrays.toString(Arrays.copyOf(GROUND_TRUTH_IDS, 15)) + "...");
                    System.out.println("  Got:      " + Arrays.toString(Arrays.copyOf(tokens, Math.min(15, tokens.length))) + "...");
                }
            }

            System.out.println("\n" + "=".repeat(60));
            System.out.println("ALL BATCH TESTS DONE");
            System.out.println("=".repeat(60));

            model.close();
        }
    }

    /**
     * Create a random RGB Mat (deterministic per call with same seed would need RNG,
     * but for testing, noise content is fine).
     */
    private static Mat createRandomRgbMat(MatManager matManager, int rows, int cols) {
        Mat mat = matManager.newMat(rows, cols, CvType.CV_8UC3);
        byte[] data = new byte[rows * cols * 3];
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) ((i * 37 + 13) % 256);
        }
        mat.put(0, 0, data);
        return mat;
    }

    /**
     * Create a filled (constant color) RGB Mat for deterministic results.
     */
    private static Mat createFilledRgbMat(MatManager matManager, int rows, int cols, int value) {
        Mat mat = matManager.newMat(rows, cols, CvType.CV_8UC3);
        byte[] data = new byte[rows * cols * 3];
        byte v = (byte) (value & 0xFF);
        Arrays.fill(data, v);
        mat.put(0, 0, data);
        return mat;
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
