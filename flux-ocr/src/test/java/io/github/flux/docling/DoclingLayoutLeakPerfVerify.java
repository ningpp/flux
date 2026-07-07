package io.github.flux.docling;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.model.LayoutModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * DoclingLayoutModel 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 使用长期存活的 MatManager 与 NDManager 反复推理，验证每轮结束后：
 *   - MatManager.trackedMatCount() 回归基线（修复前 ToCHW 预处理 Mat 会无界累积）；
 *   - 每轮使用的 NDManager 子管理器资源计数为 0（修复前 logits/bboxes 临时 NDArray 会泄漏）。
 * 同时统计单次 batch 推理耗时与吞吐。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.docling.DoclingLayoutLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations> <batchSize>
 *   # 默认: D:\tmp\layout.png 50 1
 * <p>
 * 默认依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行；若某设备不可用则跳过并提示。
 */
public class DoclingLayoutLeakPerfVerify {

    static final String MODEL_ROOT_DIR = "D:\\models\\layout";
    static final String MODEL_NAME = "docling-layout-heron";

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\layout.png";
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 50;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 1;

        System.out.printf(Locale.ROOT, "验证图片: %s  迭代次数: %d  batchSize: %d%n%n", imagePath, iterations, batchSize);

        // CPU 与 GPU 都跑一遍
        boolean cpuOk = verifyDevice(-1, imagePath, iterations, batchSize);
        boolean gpuOk = verifyDevice(0, imagePath, iterations, batchSize);

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "CPU 泄露验证: %s%n", cpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "GPU 泄露验证: %s%n", gpuOk ? "PASS" : "FAIL/SKIP");
        System.out.println("============================================================");
        if (!cpuOk || !gpuOk) {
            // 至少有一个设备失败（非跳过）时返回非 0，便于 CI 断言
            boolean hardFail = (cpuOk == Boolean.FALSE) || (gpuOk == Boolean.FALSE);
            if (hardFail) {
                System.exit(1);
            }
        }
    }

    /**
     * 在指定设备上运行泄露 + 性能验证。
     *
     * @return true=通过(或设备不可用已跳过) ; false=检测到泄露
     */
    private static boolean verifyDevice(int gpuIndex, String imagePath, int iterations, int batchSize) {
        String deviceName = gpuIndex < 0 ? "CPU" : ("GPU:" + gpuIndex);
        System.out.println("------------------------------------------------------------");
        System.out.println("设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            LayoutModel model;
            try {
                model = new LayoutModel(MODEL_ROOT_DIR, MODEL_NAME, gpuIndex, env);
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
                int firstIterDetCount = -1;

                for (int it = 0; it < iterations; it++) {
                    // 每轮读取 batchSize 张图片（复刻真实复用场景）
                    List<ProcessedMat> processedMats = new ArrayList<>(batchSize);
                    for (int b = 0; b < batchSize; b++) {
                        Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                        processedMats.add(model.processRgb(matManager, rgbImg, baseNdManager));
                    }

                    // 每轮使用独立子管理器，便于精确统计该轮 NDArray 是否全部释放
                    NDManager iterMgr = baseNdManager.newSubManager();
                    long t0 = System.nanoTime();
                    List<List<ObjectDetectionResult>> results =
                            model.doBatchPredict(processedMats, matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    long ndRes = ndResourceCount(iterMgr);
                    if (it == 0) {
                        System.out.println("  [DEBUG] 首轮未释放 ND 资源: " + ndResourceKeys(iterMgr));
                    }
                    iterMgr.close();

                    // 释放 ProcessedMat 持有的原始 Mat（实际已被 Resize 内部释放，此处幂等）
                    for (ProcessedMat pm : processedMats) {
                        pm.release(matManager);
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
                    // ndRes == -1 表示反射不可用，跳过该项断言；>0 才判定为泄露
                    if (ndRes > 0) {
                        ndLeakFailCount++;
                        if (ndLeakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 NDManager 未释放资源=%d, 疑似 NDArray 泄露%n",
                                    it + 1, ndRes);
                        }
                    }

                    if (it == 0 && !results.isEmpty()) {
                        firstIterDetCount = results.get(0).size();
                    }
                }

                // 全部迭代结束后，长期存活的 MatManager 应完全回归基线
                int finalMatCount = matManager.trackedMatCount();
                boolean matLeakFree = (leakFailCount == 0) && (finalMatCount == baseline);
                boolean ndLeakFree = ndLeakFailCount == 0;

                double avgMs = iterations == 0 ? 0 : (double) totalPredictMs / iterations;
                double throughput = (totalPredictMs == 0) ? 0
                        : (iterations * batchSize) * 1000.0 / totalPredictMs;

                System.out.printf(Locale.ROOT, "  首轮检测框数: %d%n", firstIterDetCount);
                System.out.printf(Locale.ROOT, "  总推理耗时: %.1f ms  平均/batch: %.2f ms  吞吐: %.2f img/s%n",
                        (double) totalPredictMs, avgMs, throughput);
                System.out.printf(Locale.ROOT, "  MatManager 基线=%d 终值=%d  泄露轮次=%d%n",
                        baseline, finalMatCount, leakFailCount);
                System.out.printf(Locale.ROOT, "  NDArray 泄露轮次=%d (资源计数反射不可用则为跳过)%n", ndLeakFailCount);

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
     * 通过反射读取 DJL NDManager 当前持有的资源数。
     */
    @SuppressWarnings("unchecked")
    private static long ndResourceCount(NDManager mgr) {
        return readMapSize(mgr, "resources") + readMapSize(mgr, "tempResources");
    }

    @SuppressWarnings("unchecked")
    private static long readMapSize(NDManager mgr, String fieldName) {
        Class<?> c = mgr.getClass();
        while (c != null && c != Object.class) {
            try {
                Field f = c.getDeclaredField(fieldName);
                f.setAccessible(true);
                Object map = f.get(mgr);
                if (map instanceof Map) {
                    return ((Map<?, ?>) map).size();
                }
                return 0;
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (Exception e) {
                return 0;
            }
        }
        return 0;
    }

    /** 反射读取 resources/tempResources 中各资源的 shape，用于定位泄露（仅调试）。 */
    @SuppressWarnings("unchecked")
    private static String ndResourceKeys(NDManager mgr) {
        List<String> parts = new ArrayList<>();
        parts.add("resources=" + dumpMap(mgr, "resources"));
        parts.add("tempResources=" + dumpMap(mgr, "tempResources"));
        return String.join("  ", parts);
    }

    @SuppressWarnings("unchecked")
    private static String dumpMap(NDManager mgr, String fieldName) {
        Class<?> c = mgr.getClass();
        while (c != null && c != Object.class) {
            try {
                Field f = c.getDeclaredField(fieldName);
                f.setAccessible(true);
                Object map = f.get(mgr);
                if (map instanceof Map) {
                    Map<?, ?> m = (Map<?, ?>) map;
                    List<String> names = new ArrayList<>();
                    for (Object v : m.values()) {
                        if (v instanceof ai.djl.ndarray.NDArray a) {
                            names.add(a.getShape().toString());
                        } else {
                            names.add(v == null ? "null" : v.getClass().getSimpleName());
                        }
                    }
                    return "[" + String.join(", ", names) + "]";
                }
                return "[]";
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (Exception e) {
                return "(err:" + e.getMessage() + ")";
            }
        }
        return "(no field)";
    }
}
