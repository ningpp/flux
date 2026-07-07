package io.github.flux.formula;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Pix2Text 公式识别 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 验证目标（本次修复点）：
 *   1) DeiTImageProcessor.preprocess 链式算子（resize/rescale/normalize/transpose 及
 *      NDArrays.stack 的输入）在每轮结束后全部释放，不再随迭代累积；
 *   2) 每轮使用的 NDManager 子管理器资源计数为 0（验证不再泄漏 DJL NDArray）；
 *   3) MatManager.trackedMatCount() 回归基线；
 *   4) 解码结果与修复前一致（正确性），并打印单次推理耗时。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.formula.Pix2TextFormulaLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations>
 *   # 默认: D:\tmp\formula-2026-01-18-152316.png 50
 * <p>
 * 默认依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行。
 */
public class Pix2TextFormulaLeakPerfVerify {

    static final String MODEL_ROOT_DIR = "D:\\models\\formula";
    static final String MODEL_NAME = "pix2text-mfr-1.5";

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\formula-2026-01-18-152316.png";
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 50;

        System.out.printf(Locale.ROOT, "验证图片: %s  迭代次数: %d  模型: %s%n%n", imagePath, iterations, MODEL_NAME);

        boolean cpuOk = verifyDevice(-1, imagePath, iterations);
        boolean gpuOk = verifyDevice(0, imagePath, iterations);

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "CPU 泄露验证: %s%n", cpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "GPU 泄露验证: %s%n", gpuOk ? "PASS" : "FAIL/SKIP");
        System.out.println("============================================================");
        if (cpuOk == Boolean.FALSE || gpuOk == Boolean.FALSE) {
            System.exit(1);
        }
    }

    private static boolean verifyDevice(int gpuIndex, String imagePath, int iterations) {
        String deviceName = gpuIndex < 0 ? "CPU" : ("GPU:" + gpuIndex);
        System.out.println("------------------------------------------------------------");
        System.out.println("设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            FormulaRecognitionModel model;
            try {
                model = new FormulaRecognitionModel(MODEL_ROOT_DIR, MODEL_NAME, gpuIndex, env);
            } catch (Exception e) {
                System.out.println("  [SKIP] 设备不可用，无法创建模型: " + e.getMessage());
                return true;
            }

            try (var baseNdManager = NDManager.newBaseManager();
                 var matManager = new MatManager()) {

                int baseline = matManager.trackedMatCount();
                long totalPredictMs = 0;
                int matLeakFailCount = 0;
                int ndLeakFailCount = 0;
                String firstDecoded = null;

                for (int it = 0; it < iterations; it++) {
                    // 每轮使用独立子管理器，便于精确统计该轮 NDArray 是否全部释放
                    NDManager iterMgr = baseNdManager.newSubManager();

                    Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                    PreProcessResult ppr = model.processRgb(matManager, rgbImg, iterMgr);

                    long t0 = System.nanoTime();
                    List<TextResult> results = model.doBatchPredict(List.of(ppr), matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    long ndRes = ndResourceCount(iterMgr);
                    if (it == 0) {
                        System.out.println("  [DEBUG] 首轮未释放 ND 资源: " + ndResourceKeys(iterMgr));
                        if (!results.isEmpty()) {
                            firstDecoded = results.get(0).text();
                        }
                    }
                    iterMgr.close();

                    int matCount = matManager.trackedMatCount();
                    if (matCount != baseline) {
                        matLeakFailCount++;
                        if (matLeakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 MatManager 计数=%d (基线=%d), 疑似 Mat 泄露%n",
                                    it + 1, matCount, baseline);
                        }
                    }
                    if (ndRes > 0) {
                        ndLeakFailCount++;
                        if (ndLeakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 NDManager 未释放资源=%d, 疑似 NDArray 泄露%n",
                                    it + 1, ndRes);
                        }
                    }

                    if (it == 0 && !results.isEmpty()) {
                        System.out.printf(Locale.ROOT, "  首轮解码结果: %s%n", firstDecoded);
                        System.out.printf(Locale.ROOT, "  首轮 token 数: %d%n", results.get(0).tokens().length);
                    }
                }

                int finalMatCount = matManager.trackedMatCount();
                boolean matLeakFree = (matLeakFailCount == 0) && (finalMatCount == baseline);
                boolean ndLeakFree = ndLeakFailCount == 0;

                double avgMs = iterations == 0 ? 0 : (double) totalPredictMs / iterations;
                System.out.printf(Locale.ROOT, "  总推理耗时: %.1f ms  平均/次: %.2f ms%n",
                        (double) totalPredictMs, avgMs);
                System.out.printf(Locale.ROOT, "  MatManager 基线=%d 终值=%d  泄露轮次=%d%n",
                        baseline, finalMatCount, matLeakFailCount);
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

    @SuppressWarnings("unchecked")
    private static String ndResourceKeys(NDManager mgr) {
        return "resources=" + dumpMap(mgr, "resources") + "  tempResources=" + dumpMap(mgr, "tempResources");
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
                    java.util.List<String> names = new java.util.ArrayList<>();
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
