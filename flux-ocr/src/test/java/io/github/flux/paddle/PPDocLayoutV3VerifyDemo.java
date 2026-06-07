package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.model.LayoutModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Verifies PP-DocLayoutV3 Java inference results against the Python reference output.
 * Uses content-based matching (not positional) to handle minor float precision differences
 * in reading order computation.
 */
public class PPDocLayoutV3VerifyDemo {

    static final String MODEL_ROOT_DIR = "D:\\models\\layout";
    static final String MODEL_NAME = "PP-DocLayoutV3";
    static final List<String> TEST_IMAGES = List.of(
            "d:\\code\\pp-doclayoutv3-convert\\imgs\\deepseek-v4-26-6-7-1132.png",
            "d:\\code\\pp-doclayoutv3-convert\\imgs\\layout-2026-06-07-113015.png"
    );

    // Python reference: (label, score, [x1, y1, x2, y2])
    static final List<RefDet> IMAGE1_REF = List.of(
            new RefDet("chart",           0.926f, new float[]{75.4f,  559.1f, 381.1f, 772.2f}),
            new RefDet("chart",           0.915f, new float[]{416.1f, 528.4f, 586.9f, 662.7f}),
            new RefDet("header_image",    0.892f, new float[]{77.4f,  38.1f,  217.8f, 68.6f}),
            new RefDet("abstract",        0.932f, new float[]{74.0f,  241.7f, 591.4f, 517.9f}),
            new RefDet("chart",           0.911f, new float[]{418.5f, 669.6f, 586.6f, 804.2f}),
            new RefDet("text",            0.816f, new float[]{259.5f, 175.6f, 402.0f, 207.3f}),
            new RefDet("paragraph_title", 0.802f, new float[]{296.6f, 216.8f, 368.4f, 234.5f}),
            new RefDet("doc_title",       0.570f, new float[]{272.0f, 127.5f, 395.0f, 147.6f}),
            new RefDet("figure_title",    0.918f, new float[]{73.0f,  815.1f, 591.1f, 845.8f}),
            new RefDet("doc_title",       0.689f, new float[]{79.7f,  150.1f, 583.8f, 171.3f})
    );

    static final List<RefDet> IMAGE2_REF = List.of(
            new RefDet("image",           0.972f, new float[]{58.3f,  16.8f,  279.0f, 179.3f}),
            new RefDet("text",            0.954f, new float[]{284.7f, 107.0f, 510.2f, 200.0f}),
            new RefDet("text",            0.950f, new float[]{55.0f,  269.3f, 280.1f, 351.2f}),
            new RefDet("text",            0.967f, new float[]{284.6f, 280.5f, 510.7f, 443.0f}),
            new RefDet("text",            0.956f, new float[]{55.2f,  361.3f, 280.0f, 431.9f}),
            new RefDet("image",           0.947f, new float[]{59.8f,  451.4f, 509.2f, 575.1f}),
            new RefDet("text",            0.946f, new float[]{285.1f, 210.9f, 510.5f, 269.6f}),
            new RefDet("figure_title",    0.888f, new float[]{55.3f,  187.0f, 280.5f, 210.2f}),
            new RefDet("text",            0.912f, new float[]{54.9f,  220.8f, 279.9f, 245.7f}),
            new RefDet("paragraph_title", 0.810f, new float[]{56.1f,  255.6f, 171.4f, 267.1f}),
            new RefDet("figure_title",    0.867f, new float[]{286.7f, 83.7f,  501.2f, 95.0f}),
            new RefDet("figure_title",    0.907f, new float[]{56.2f,  581.7f, 510.5f, 604.0f}),
            new RefDet("image",           0.776f, new float[]{289.1f, 17.3f,  508.9f, 77.4f}),
            new RefDet("number",          0.782f, new float[]{482.9f, 649.2f, 509.8f, 659.7f})
    );

    static final List<List<RefDet>> ALL_REFS = List.of(IMAGE1_REF, IMAGE2_REF);
    static final int[] EXPECTED_COUNTS = {10, 14};

    record RefDet(String label, float score, float[] box) {}

    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("PP-DocLayoutV3 Java vs Python 验证");
        System.out.println("============================================================\n");

        try (var env = OrtEnvironment.getEnvironment();
             var model = new LayoutModel(MODEL_ROOT_DIR, MODEL_NAME, -1, env);
             var ndManager = NDManager.newBaseManager();
             var matManager = new MatManager()) {

            List<ProcessedMat> processedMats = new ArrayList<>();
            for (String imgPath : TEST_IMAGES) {
                Mat rgbImg = ImageUtil.readToRgb(matManager, imgPath);
                ProcessedMat pm = model.processRgb(matManager, rgbImg, ndManager);
                processedMats.add(pm);
            }

            List<List<ObjectDetectionResult>> allResults =
                    model.doBatchPredict(processedMats, matManager, ndManager, Map.of());

            boolean allPass = true;
            for (int i = 0; i < allResults.size(); i++) {
                List<ObjectDetectionResult> javaResults = allResults.get(i);
                String imgName = TEST_IMAGES.get(i).substring(TEST_IMAGES.get(i).lastIndexOf('\\') + 1);
                List<RefDet> refs = ALL_REFS.get(i);
                int expectedCount = EXPECTED_COUNTS[i];

                System.out.println("--- 图片 " + (i + 1) + ": " + imgName + " ---");
                System.out.println("  Java 检测框数: " + javaResults.size() + "  |  Python 期望: " + expectedCount);

                // 1. Check detection count
                boolean countMatch = javaResults.size() == expectedCount;
                if (!countMatch) {
                    System.out.println("  [FAIL] 检测框数量不一致!");
                    allPass = false;
                } else {
                    System.out.println("  [PASS] 检测框数量一致");
                }

                // 2. Content-based matching: for each Java result, find the best matching Python ref
                boolean[] refUsed = new boolean[refs.size()];
                float maxScoreDiff = 0;
                float maxBoxDiff = 0;
                int labelMismatchCount = 0;
                int matchedCount = 0;

                for (ObjectDetectionResult javaResult : javaResults) {
                    int bestRefIdx = -1;
                    float bestDist = Float.MAX_VALUE;
                    for (int r = 0; r < refs.size(); r++) {
                        if (refUsed[r]) continue;
                        RefDet ref = refs.get(r);
                        if (!javaResult.label().equals(ref.label)) continue;
                        float boxDist = 0;
                        for (int k = 0; k < 4; k++) {
                            boxDist += Math.abs(javaResult.coordinate()[k] - ref.box[k]);
                        }
                        if (boxDist < bestDist) {
                            bestDist = boxDist;
                            bestRefIdx = r;
                        }
                    }
                    if (bestRefIdx >= 0) {
                        refUsed[bestRefIdx] = true;
                        matchedCount++;
                        RefDet ref = refs.get(bestRefIdx);
                        float scoreDiff = Math.abs(javaResult.score() - ref.score);
                        float boxDiff = 0;
                        for (int k = 0; k < 4; k++) {
                            boxDiff = Math.max(boxDiff, Math.abs(javaResult.coordinate()[k] - ref.box[k]));
                        }
                        maxScoreDiff = Math.max(maxScoreDiff, scoreDiff);
                        maxBoxDiff = Math.max(maxBoxDiff, boxDiff);
                    } else {
                        // No matching label found — try matching by box proximity
                        for (int r = 0; r < refs.size(); r++) {
                            if (refUsed[r]) continue;
                            RefDet ref = refs.get(r);
                            float boxDist = 0;
                            for (int k = 0; k < 4; k++) {
                                boxDist += Math.abs(javaResult.coordinate()[k] - ref.box[k]);
                            }
                            if (boxDist < bestDist) {
                                bestDist = boxDist;
                                bestRefIdx = r;
                            }
                        }
                        if (bestRefIdx >= 0) {
                            refUsed[bestRefIdx] = true;
                            matchedCount++;
                            labelMismatchCount++;
                            RefDet ref = refs.get(bestRefIdx);
                            System.out.printf(Locale.ROOT, "  [WARN] label mismatch: java=%s ref=%s box=[%.1f,%.1f,%.1f,%.1f]%n",
                                    javaResult.label(), ref.label,
                                    javaResult.coordinate()[0], javaResult.coordinate()[1],
                                    javaResult.coordinate()[2], javaResult.coordinate()[3]);
                        }
                    }
                }

                // 3. Summary
                boolean countOk = matchedCount == expectedCount && labelMismatchCount == 0;
                boolean scoreOk = maxScoreDiff < 0.01f;
                boolean boxOk = maxBoxDiff < 2.0f;
                boolean pass = countOk && scoreOk && boxOk;
                if (!pass) allPass = false;

                String status = pass ? "PASS" : "FAIL";
                System.out.printf(Locale.ROOT, "  [%s] 匹配数=%d/%d  标签不匹配=%d  最大分数差=%.6f  最大坐标差=%.4f%n",
                        status, matchedCount, expectedCount, labelMismatchCount, maxScoreDiff, maxBoxDiff);

                // 4. Print all Java results
                System.out.println("  --- Java 结果 (按阅读顺序) ---");
                for (int j = 0; j < javaResults.size(); j++) {
                    ObjectDetectionResult r = javaResults.get(j);
                    System.out.printf(Locale.ROOT, "    [%2d] %-20s score=%.3f  box=[%.1f,%.1f,%.1f,%.1f]%n",
                            j + 1, r.label(), r.score(),
                            r.coordinate()[0], r.coordinate()[1],
                            r.coordinate()[2], r.coordinate()[3]);
                }
                System.out.println();
            }

            System.out.println("============================================================");
            if (allPass) {
                System.out.println("全部验证通过！Java 推理结果与 Python 参考结果一致。");
            } else {
                System.out.println("验证未通过，请检查 Java 后处理逻辑。");
            }
            System.out.println("============================================================");
        }
    }
}
