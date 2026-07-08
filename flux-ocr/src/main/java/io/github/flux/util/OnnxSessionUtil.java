package io.github.flux.util;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * 统一的 ONNX 会话创建工具。
 *
 * <p>集中处理两个系统性 GPU 内存问题（OCRPipelineGPUPerfTest 观测到的严重显存增长根因）：
 * <ol>
 *   <li>CUDA 会话默认开启 {@code memoryPatternOptimization}，会对<b>每种不同输入 shape</b>
 *       缓存一份内存分配方案，且该缓存在会话生命周期内不回收。OCR 每页 pad 后尺寸不同，
 *       导致 GPU 显存随页面尺寸种类无界累积。这里显式关闭它。</li>
 *   <li>{@link OrtSession.SessionOptions} 实现了 {@link AutoCloseable}，创建后必须
 *       {@code close()} 以释放原生 {@code OrtSessionOptions} 结构体，否则造成原生内存泄露。
 *       这里在 {@code createSession} 之后立即关闭。</li>
 * </ol>
 */
public final class OnnxSessionUtil {

    private OnnxSessionUtil() {
    }

    public static OrtSession createSession(OrtEnvironment env, String modelFile, int gpuIndex)
            throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try {
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            // 关闭内存模式优化，避免按输入 shape 缓存分配方案导致 GPU 显存无界增长。
            options.setMemoryPatternOptimization(false);
            return env.createSession(modelFile, options);
        } finally {
            // SessionOptions 在 createSession 后即完成使命，立即释放其原生资源。
            IOUtil.close(options);
        }
    }
}
