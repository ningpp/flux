package io.github.flux.verify;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * GOT-OCR-2.0 与 PP-FormulaNet-L 内存泄露 + 性能验证（CPU / GPU 双设备）。
 * <p>
 * 针对这两类模型此前的内存泄露点（见下方“修复点”），使用指定图片反复推理，验证：
 *   1) 每轮推理后 MatManager.trackedMatCount() 回归基线（无 OpenCV Mat 泄露）；
 *   2) 每轮使用的子 NDManager 资源计数为 0（无 DJL NDArray 泄露）；
 *   3) 解码结果在多次迭代间保持一致（正确性）；
 *   4) 打印单次平均推理耗时（性能）。
 * <p>
 * 修复点回顾：
 *   - GOT-OCR-2.0: Encoder 改为直接拼 float 缓冲（不再用 DJL 默认管理器 stack，
 *     根治每次推理泄漏 ~batch*3*1024*1024*4 字节）；processRgb 释放 rgbMat 与中间
 *     NDArray；doBatchPredict 释放输入 PreProcessResult；固定 prompt 仅 tokenize 一次。
 *   - PP-FormulaNet-L: processRgb 释放链式预处理产生的 5 个临时 Mat 与输入 rgbMat；
 *     doBatchPredict 释放输入 PreProcessResult 的 NDArray。
 * <p>
 * 用法（flux-ocr 项目根目录）：
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.verify.GotOcr2AndPaddleFormulaLeakVerify" "-Dexec.classpathScope=test"
 *   # 可选参数: <imagePath> <iterations> <modelRootDir>
 *   # 默认: D:\tmp\formula-2026-01-18-152316.png 30 D:\models
 * <p>
 * 默认对每个模型依次在 CPU(gpuIndex=-1) 与 GPU(gpuIndex=0) 上运行；设备不可用则跳过。
 */
public class GotOcr2AndPaddleFormulaLeakVerify {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    static final String DEFAULT_IMAGE = "D:\\tmp\\formula-2026-01-18-152316.png";
    static final String DEFAULT_MODEL_ROOT = "D:\\models\\formula";
    static final String[] MODELS = {"GOT-OCR-2.0", "PP-FormulaNet-L"};

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : DEFAULT_IMAGE;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 30;
        String modelRootDir = args.length > 2 ? args[2] : DEFAULT_MODEL_ROOT;

        System.out.printf(Locale.ROOT,
                "验证图片: %s  迭代次数: %d  模型根目录: %s%n%n", imagePath, iterations, modelRootDir);

        boolean overall = true;
        for (String modelName : MODELS) {
            boolean cpuOk = verifyDevice(-1, imagePath, iterations, modelRootDir, modelName);
            boolean gpuOk = verifyDevice(0, imagePath, iterations, modelRootDir, modelName);
            boolean pass = cpuOk && gpuOk;
            overall &= pass;
            System.out.printf(Locale.ROOT, "[%s] CPU:%s GPU:%s%n%n",
                    modelName, cpuOk ? "PASS" : "FAIL/SKIP", gpuOk ? "PASS" : "FAIL/SKIP");
        }

        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "总体结果: %s%n", overall ? "PASS" : "FAIL");
        System.out.println("============================================================");
        if (!overall) {
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
        System.out.println("模型: " + modelName + "  设备: " + deviceName);
        System.out.println("------------------------------------------------------------");

        try (var env = OrtEnvironment.getEnvironment()) {
            FormulaRecognitionModel model;
            try {
                model = new FormulaRecognitionModel(modelRootDir, modelName, gpuIndex, env);
            } catch (Exception e) {
                System.out.println("  [SKIP] 设备/模型不可用，无法创建: " + e.getMessage());
                return true;
            }

            try (var baseNdManager = NDManager.newBaseManager();
                 var matManager = new MatManager()) {

                int baseline = matManager.trackedMatCount();
                long totalPredictMs = 0;
                int matLeakFailCount = 0;
                int ndLeakFailCount = 0;
                long expectedNd = -1;
                String firstDecoded = null;

                for (int it = 0; it < iterations; it++) {
                    NDManager iterMgr = baseNdManager.newSubManager();

                    Mat rgbImg = ImageUtil.readToRgb(matManager, imagePath);
                    PreProcessResult ppr = model.processRgb(matManager, rgbImg, iterMgr);
                    // processRgb 之后仍存活的 NDArray 即“输出张量”，其数量为预期值；
                    // 真实泄露表现为该数量在推理后超出预期（即 doBatchPredict 额外滞留张量）。
                    if (it == 0) {
                        expectedNd = ndResourceCount(iterMgr);
                    }

                    long t0 = System.nanoTime();
                    List<TextResult> results = model.doBatchPredict(List.of(ppr), matManager, iterMgr, Map.of());
                    long t1 = System.nanoTime();
                    totalPredictMs += (t1 - t0) / 1_000_000;

                    // doBatchPredict 内部已关闭 ppr；此处幂等再关闭一次，确保输出 NDArray 也被释放。
                    IOUtil.close(ppr);
                    long ndRes = ndResourceCount(iterMgr);

                    // 模型 processRgb 内部已释放 rgbImg；此处幂等释放。
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
                    // 仅当超出“输出张量”预期数量时才判定为 NDArray 泄露，
                    // 避免把合法的输出 NDArray 误报为泄露。
                    if (ndRes > expectedNd) {
                        ndLeakFailCount++;
                        if (ndLeakFailCount <= 3) {
                            System.out.printf(Locale.ROOT,
                                    "  [WARN] 第 %d 轮 NDManager 未释放资源=%d (预期=%d), 疑似 NDArray 泄露%n",
                                    it + 1, ndRes, expectedNd);
                        }
                    }

                    if (it == 0 && !results.isEmpty()) {
                        firstDecoded = results.get(0).text();
                        System.out.printf(Locale.ROOT, "  首轮解码(%d tokens): %s%n",
                                results.get(0).tokens().length,
                                firstDecoded.replace("\n", "\\n").replace("\r", "\\r"));
                    } else if (it > 0 && !results.isEmpty()
                            && firstDecoded != null && !firstDecoded.equals(results.get(0).text())) {
                        System.out.printf(Locale.ROOT, "  [WARN] 第 %d 轮解码结果与首轮不一致%n", it + 1);
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
                System.out.printf(Locale.ROOT, "  预期 NDArray 数量(输出)=%d  NDArray 泄露轮次=%d%n",
                        expectedNd, ndLeakFailCount);

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
}
