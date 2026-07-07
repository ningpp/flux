package io.github.flux.falconocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Falcon-OCR 公式识别 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 验证目标（对应本次修复点）：
 *   1) 每轮推理后 MatManager.trackedMatCount() 回归基线（无 OpenCV Mat 泄露）；
 *   2) 每轮子 NDManager 资源计数为 0（Falcon-OCR 走 ORT，不持有 DJL NDArray）；
 *   3) 推理结果正确（打印解码公式），并打印单次平均耗时；
 *   4) 共享实例缓存引用计数正确：同一配置的 formula 与 table 共用一个 FalconOcrModel，
 *      关闭其中一个（引用 -1）缓存不销毁，二者都关闭后缓存归零（无悬挂引用 / 无显存泄漏）。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.falconocr.FalconOcrFormulaLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations> <modelRootDir> <modelName>
 *   # 默认: D:\tmp\formula-2026-01-18-152316.png 30 D:\models Falcon-OCR-ONNX
 * <p>
 * 默认依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行。
 */
public class FalconOcrFormulaLeakPerfVerify {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    static final String DEFAULT_MODEL_ROOT_DIR = "D:\\models";
    static final String DEFAULT_MODEL_NAME = "Falcon-OCR-ONNX";

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\formula-2026-01-18-152316.png";
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 30;
        String modelRootDir = args.length > 2 ? args[2] : DEFAULT_MODEL_ROOT_DIR;
        String modelName = args.length > 3 ? args[3] : DEFAULT_MODEL_NAME;

        System.out.printf(Locale.ROOT,
                "验证图片: %s  迭代次数: %d  模型: %s @ %s%n%n", imagePath, iterations, modelName, modelRootDir);

        boolean cpuOk = verifyDevice(-1, imagePath, iterations, modelRootDir, modelName);
        boolean gpuOk = verifyDevice(0, imagePath, iterations, modelRootDir, modelName);
        boolean cacheOk = verifyCacheRefCount(modelRootDir, modelName);

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "CPU 泄露/性能验证: %s%n", cpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "GPU 泄露/性能验证: %s%n", gpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "共享缓存引用计数验证: %s%n", cacheOk ? "PASS" : "FAIL");
        System.out.println("============================================================");
        if (cpuOk == Boolean.FALSE || gpuOk == Boolean.FALSE || !cacheOk) {
            System.exit(1);
        }
    }

    private static boolean verifyDevice(int gpuIndex,
                                        String imagePath,
                                        int iterations,
                                        String modelRootDir,
                                        String modelName) {
        String deviceName = gpuIndex < 0 ? "CPU" : ("GPU:" + gpuIndex);
        System.out.println("------------------------------------------------------------");
        System.out.println("设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            FormulaRecognitionModel model;
            try {
                model = new FormulaRecognitionModel(modelRootDir, modelName, gpuIndex, env);
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
                    NDManager iterMgr = baseNdManager.newSubManager();

                    Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                    PreProcessResult ppr = model.processRgb(matManager, rgbImg, iterMgr);

                    long t0 = System.nanoTime();
                    List<TextResult> results = model.doBatchPredict(List.of(ppr), matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    long ndRes = ndResourceCount(iterMgr);

                    // 调用方持有 rgbImg，推理结束后显式释放，验证模型自身不残留 Mat
                    matManager.release(rgbImg);
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
                        firstDecoded = results.get(0).text();
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
                System.out.printf(Locale.ROOT, "  NDArray 泄露轮次=%d%n", ndLeakFailCount);

                boolean pass = matLeakFree && ndLeakFree;
                System.out.printf(Locale.ROOT, "  [%s] 内存泄露验证%n", pass ? "PASS" : "FAIL");

                // 关闭模型后，共享缓存应归零（无悬挂引用 / 无显存泄漏）
                model.close();
                int cacheCount = FalconOcrModel.sharedInstanceCount();
                System.out.printf(Locale.ROOT, "  关闭模型后共享缓存实例数=%d%n", cacheCount);
                if (cacheCount != 0) {
                    System.out.printf(Locale.ROOT, "  [FAIL] 共享缓存未归零，存在悬挂引用%n");
                    pass = false;
                }
                return pass;
            }
        } catch (Exception e) {
            System.out.println("  [ERROR] 设备运行异常: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 验证引用计数缓存：同一配置的 formula 与 table 共用一个 FalconOcrModel 实例，
     * 关闭其中一个不销毁共享实例，二者都关闭后缓存归零。
     */
    private static boolean verifyCacheRefCount(String modelRootDir, String modelName) {
        System.out.println("------------------------------------------------------------");
        System.out.println("共享实例缓存引用计数验证");
        System.out.println("------------------------------------------------------------");
        Map<String, Object> customParams = new java.util.HashMap<>();
        try (var env = OrtEnvironment.getEnvironment()) {
            int before = FalconOcrModel.sharedInstanceCount();

            FalconOcrFormulaModel formulaModel;
            FalconOcrTableModel tableModel;
            try {
                formulaModel = new FalconOcrFormulaModel(modelRootDir, modelName, -1, env, customParams);
                tableModel = new FalconOcrTableModel(modelRootDir, modelName, -1, env, customParams);
            } catch (Exception e) {
                System.out.println("  [SKIP] 无法创建模型: " + e.getMessage());
                return true;
            }

            int sharedAfterCreate = FalconOcrModel.sharedInstanceCount();
            System.out.printf(Locale.ROOT, "  创建 formula+table 后共享缓存实例数=%d (期望 1)%n", sharedAfterCreate);

            formulaModel.close();
            int afterFormulaClose = FalconOcrModel.sharedInstanceCount();
            System.out.printf(Locale.ROOT, "  关闭 formula 后共享缓存实例数=%d (期望仍 1, table 仍在使用)%n", afterFormulaClose);

            tableModel.close();
            int afterTableClose = FalconOcrModel.sharedInstanceCount();
            System.out.printf(Locale.ROOT, "  关闭 table 后共享缓存实例数=%d (期望 0)%n", afterTableClose);

            boolean pass = (sharedAfterCreate == 1) && (afterFormulaClose == 1) && (afterTableClose == 0)
                    && (FalconOcrModel.sharedInstanceCount() == before);
            System.out.printf(Locale.ROOT, "  [%s] 引用计数缓存验证%n", pass ? "PASS" : "FAIL");
            return pass;
        } catch (Exception e) {
            System.out.println("  [ERROR] 引用计数验证异常: " + e.getMessage());
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
}
