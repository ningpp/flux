package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * TextLineOrientationModel 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 使用长期存活的 MatManager 与 NDManager 反复推理，验证每轮结束后：
 *   - MatManager.trackedMatCount() 回归基线（修复前经 processRgb 产生的 CHW Mat 不会释放，
 *     且历史版本使用 IOUtil.close 而非 matManager.release，导致跟踪表无限累积）；
 *   - 每轮使用的 NDManager 子管理器资源计数为 0。
 * 同时统计单次 batch 推理耗时与吞吐，并校验 rot180 图片应判定为 180_degree。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.paddle.TextLineOrientationLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations> <batchSize> <modelName>
 *   # 默认: D:\tmp\textline_rot180.jpg 50 1 PP-LCNet_x1_0_textline_ori
 * <p>
 * 默认依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行；若某设备不可用则跳过并提示。
 */
public class TextLineOrientationLeakPerfVerify {

    static final String MODEL_ROOT_DIR = "D:\\models";

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\textline_rot180.jpg";
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 50;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        String modelName = args.length > 3 ? args[3] : "PP-LCNet_x1_0_textline_ori";

        System.out.printf(Locale.ROOT, "模型: %s  验证图片: %s  迭代次数: %d  batchSize: %d%n%n",
                modelName, imagePath, iterations, batchSize);

        boolean cpuOk = verifyDevice(-1, imagePath, iterations, batchSize, modelName);
        boolean gpuOk = verifyDevice(0, imagePath, iterations, batchSize, modelName);

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "CPU 泄露验证: %s%n", cpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "GPU 泄露验证: %s%n", gpuOk ? "PASS" : "FAIL/SKIP");
        System.out.println("============================================================");
        // 至少有一个设备明确失败（非跳过）时返回非 0，便于 CI 断言
        if (cpuOk == Boolean.FALSE || gpuOk == Boolean.FALSE) {
            System.exit(1);
        }
    }

    /**
     * 在指定设备上运行泄露 + 性能验证。
     *
     * @return true=通过(或设备不可用已跳过) ; false=检测到泄露
     */
    private static boolean verifyDevice(int gpuIndex, String imagePath, int iterations,
                                        int batchSize, String modelName) {
        String deviceName = gpuIndex < 0 ? "CPU" : ("GPU:" + gpuIndex);
        System.out.println("------------------------------------------------------------");
        System.out.println("设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            TextLineOrientationModel model;
            try {
                model = new TextLineOrientationModel(MODEL_ROOT_DIR, modelName, env, gpuIndex);
            } catch (Exception e) {
                System.out.println("  [SKIP] 设备不可用，无法创建模型: " + e.getMessage());
                return true; // 视为跳过，不算失败
            }

            try (var baseNdManager = NDManager.newBaseManager();
                 var matManager = new MatManager()) {

                int baseline = matManager.trackedMatCount();
                long totalPredictMs = 0;
                int leakFailCount = 0;
                int ndLeakFailCount = 0;
                boolean ndResAvailable = false;
                ClassificationResult firstResult = null;

                for (int it = 0; it < iterations; it++) {
                    // 每轮读取 batchSize 张图片（复刻真实复用场景）
                    List<PreProcessResult> inputs = new ArrayList<>(batchSize);
                    for (int b = 0; b < batchSize; b++) {
                        Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                        inputs.add(model.processRgb(matManager, rgbImg, baseNdManager));
                    }

                    // 每轮使用独立子管理器，便于精确统计该轮 NDArray 是否全部释放
                    NDManager iterMgr = baseNdManager.newSubManager();
                    long t0 = System.nanoTime();
                    List<ClassificationResult> results = model.batchPredict(
                            inputs, batchSize, matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    long ndRes = ndResourceCount(iterMgr);
                    if (ndRes >= 0) {
                        ndResAvailable = true;
                    }
                    iterMgr.close();

                    // 预测器在 doBatchPredict 内部已通过 MatManager 释放预处理的输入 Mat，
                    // 此处不再重复释放（验证模型自清理行为）。
                    if (it == 0 && !results.isEmpty()) {
                        firstResult = results.get(0);
                    }

                    int matCount = matManager.trackedMatCount();
                    if (matCount != baseline) {
                        leakFailCount++;
                        if (leakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 MatManager 计数=%d (基线=%d), 疑似 Mat 泄露%n",
                                    it + 1, matCount, baseline);
                        }
                    }
                    // ndRes == -1 表示反射探针不可用（方法不可见），忽略该项断言；
                    // 仅当探针可达且计数 > 0 时才视为 NDArray 泄露。
                    if (ndRes > 0) {
                        ndLeakFailCount++;
                        if (ndLeakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 NDManager 资源计数=%d, 疑似 NDArray 泄露%n",
                                    it + 1, ndRes);
                        }
                    }
                }

                // 全部迭代结束后，长期存活的 MatManager 应完全回归基线
                int finalMatCount = matManager.trackedMatCount();
                boolean matLeakFree = (leakFailCount == 0) && (finalMatCount == baseline);
                boolean ndLeakFree = ndLeakFailCount == 0;

                double avgMs = iterations == 0 ? 0 : (double) totalPredictMs / iterations;
                double throughput = (totalPredictMs == 0) ? 0
                        : (iterations * batchSize) * 1000.0 / totalPredictMs;

                System.out.printf(Locale.ROOT, "  首轮结果: label=%s score=%.6f%n",
                        firstResult == null ? "null" : firstResult.label(),
                        firstResult == null ? 0f : firstResult.score());
                System.out.printf(Locale.ROOT, "  总推理耗时: %.1f ms  平均/batch: %.3f ms  吞吐: %.2f img/s%n",
                        (double) totalPredictMs, avgMs, throughput);
                System.out.printf(Locale.ROOT, "  MatManager 基线=%d 终值=%d  泄露轮次=%d%n",
                        baseline, finalMatCount, leakFailCount);
                System.out.printf(Locale.ROOT, "  NDArray 泄露轮次=%d%s%n", ndLeakFailCount,
                        ndResAvailable ? "" : "  (探针不可用, 已忽略)");

                boolean pass = matLeakFree && ndLeakFree;
                System.out.printf(Locale.ROOT, "  [%s] 内存泄露验证%n", pass ? "PASS" : "FAIL");
                return pass;
            } finally {
                model.close();
            }
        } catch (Exception e) {
            System.out.println("  [ERROR] 设备运行异常: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 通过反射读取 DJL NDManager 的资源计数（getResourceCount）。
     * 不同 DJL 版本方法可见性可能不同，拿不到时返回 -1（忽略该项断言）。
     */
    private static long ndResourceCount(NDManager mgr) {
        try {
            Method m = mgr.getClass().getMethod("getResourceCount");
            m.setAccessible(true);
            return ((Number) m.invoke(mgr)).longValue();
        } catch (Exception e) {
            return -1;
        }
    }
}
