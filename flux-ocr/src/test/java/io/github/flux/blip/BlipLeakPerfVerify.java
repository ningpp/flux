package io.github.flux.blip;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * BLIP 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 使用长期存活的 MatManager 与 NDManager 反复推理，验证每轮结束后：
 *   - MatManager.trackedMatCount() 回归基线（修复前 rgbMat / resized / floatMat 会无界累积）；
 *   - 每轮使用的 NDManager 子管理器资源计数为 0（修复前 processRgb 会把 float[] 包装成 NDArray
 *     且 doBatchPredict 取出后从不 close，导致原生 NDArray 内存泄漏）。
 * 同时统计单次 batch 推理耗时与吞吐，并打印真实 caption 文本。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.blip.BlipLeakPerfVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations> <batchSize>
 *   # 默认: D:\tmp\img-2026-02-07-120114.png 20 1
 * <p>
 * 默认依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行；若某设备不可用则跳过并提示。
 */
public class BlipLeakPerfVerify {

    static final String MODEL_ROOT_DIR = "D:\\models\\onnx";
    static final String MODEL_NAME = "blip-image-captioning-large";

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\img-2026-02-07-120114.png";
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 20;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 1;

        System.out.printf(Locale.ROOT, "验证图片: %s  迭代次数: %d  batchSize: %d%n%n", imagePath, iterations, batchSize);

        boolean cpuOk = verifyDevice(-1, imagePath, iterations, batchSize);
        boolean gpuOk = verifyDevice(0, imagePath, iterations, batchSize);

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "CPU 泄露验证: %s%n", cpuOk ? "PASS" : "FAIL/SKIP");
        System.out.printf(Locale.ROOT, "GPU 泄露验证: %s%n", gpuOk ? "PASS" : "FAIL/SKIP");
        System.out.println("============================================================");
        // 任一设备“真正失败”（非跳过）时返回非 0
        if (cpuOk == Boolean.FALSE || gpuOk == Boolean.FALSE) {
            System.exit(1);
        }
    }

    private static boolean verifyDevice(int gpuIndex, String imagePath, int iterations, int batchSize) {
        String deviceName = gpuIndex < 0 ? "CPU" : ("GPU:" + gpuIndex);
        System.out.println("------------------------------------------------------------");
        System.out.println("设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            BlipModel model;
            try {
                model = new BlipModel(MODEL_ROOT_DIR, MODEL_NAME, gpuIndex, env);
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
                String firstCaption = null;

                for (int it = 0; it < iterations; it++) {
                    // 复刻真实复用场景：每轮读取 batchSize 张图片并预处理
                    List<BlipModel.BlipPreProcessResult> pprs = new ArrayList<>(batchSize);
                    for (int b = 0; b < batchSize; b++) {
                        Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                        pprs.add(model.processRgb(matManager, rgbImg, baseNdManager));
                    }

                    // 每轮使用独立子管理器，便于精确统计该轮 NDArray 是否全部释放
                    NDManager iterMgr = baseNdManager.newSubManager();
                    long t0 = System.nanoTime();
                    List<TextResult> results = model.doBatchPredict(pprs, matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    long ndRes = ndResourceCount(iterMgr);
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
                        firstCaption = results.get(0).text();
                    }
                }

                int finalMatCount = matManager.trackedMatCount();
                boolean matLeakFree = (matLeakFailCount == 0) && (finalMatCount == baseline);
                boolean ndLeakFree = ndLeakFailCount == 0;

                double avgMs = iterations == 0 ? 0 : (double) totalPredictMs / iterations;
                double throughput = (totalPredictMs == 0) ? 0
                        : (iterations * batchSize) * 1000.0 / totalPredictMs;

                System.out.printf(Locale.ROOT, "  首轮 caption: %s%n", firstCaption);
                System.out.printf(Locale.ROOT, "  总推理耗时: %.1f ms  平均/batch: %.2f ms  吞吐: %.2f img/s%n",
                        (double) totalPredictMs, avgMs, throughput);
                System.out.printf(Locale.ROOT, "  MatManager 基线=%d 终值=%d  泄露轮次=%d%n",
                        baseline, finalMatCount, matLeakFailCount);
                System.out.printf(Locale.ROOT, "  NDArray 泄露轮次=%d (资源计数反射不可用则忽略)%n", ndLeakFailCount);

                boolean pass = matLeakFree && ndLeakFree;
                System.out.printf(Locale.ROOT, "  [%s] 内存泄露验证%n", pass ? "PASS" : "FAIL");
                return pass ? true : false;
            } finally {
                model.close();
                System.out.println("  [INFO] model.close() 执行完毕（含 tokenizer 释放），无异常");
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
}
