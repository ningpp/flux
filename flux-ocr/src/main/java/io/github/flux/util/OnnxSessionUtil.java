package io.github.flux.util;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;

import java.util.Objects;

/**
 * 统一的 ONNX 会话创建工具。
 *
 * <p>默认使用 ONNX Runtime 的原生 CUDA 配置（不修改 arena 策略、不限制显存上限、
 * 不调整 CPU arena）。在与参考 Python 实现 {@code infer_onnx.py} 的对比中发现：
 * 同一 UniRec 模型 / 图片 / 2048 token 的推理，Python 默认配置 GPU 显存峰值仅约 730 MiB；
 * 而之前 Java 中自定义的 {@code arena_extend_strategy=kSameAsRequested} 与 CPU arena 调整
 * 反而在 8 GB GPU 上触发了 Concat 节点的显存分配失败。因此这里恢复为 ORT 默认行为，
 * 让运行时按物理 GPU 显存动态分配。</p>
 *
 * <p>如仍有特殊场景需要限制 per-session CUDA 显存，可通过系统属性
 * {@code flux.ocr.onnx.cudaGpuMemLimitBytes} 手动设置；默认值为 0，表示不限制。</p>
 *
 * <p>{@link OrtSession.SessionOptions} 实现了 {@link AutoCloseable}，创建后必须
 * {@code close()} 以释放原生 {@code OrtSessionOptions} 结构体，否则造成原生内存泄露。
 * 这里在 {@code createSession} 之后立即关闭。</p>
 */
public final class OnnxSessionUtil {

    static final String CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY = "flux.ocr.onnx.cudaGpuMemLimitBytes";
    private static final long DEFAULT_CUDA_GPU_MEM_LIMIT_BYTES = 0L;

    private OnnxSessionUtil() {
    }

    public static OrtSession createSession(OrtEnvironment env, String modelFile, int gpuIndex)
            throws OrtException {
        return createSession(env, modelFile, gpuIndex, _ -> {
        });
    }

    public static OrtSession createSession(OrtEnvironment env, String modelFile, int gpuIndex,
                                           SessionOptionsConfigurer configureOptions)
            throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtCUDAProviderOptions cudaOpts = null;
        try {
            if (gpuIndex > -1) {
                cudaOpts = new OrtCUDAProviderOptions(gpuIndex);
                // Use ORT defaults. Custom arena strategies have caused GPU OOM on
                // consumer GPUs with this model family.
                long gpuMemLimitBytes = resolveCudaGpuMemLimitBytes();
                if (gpuMemLimitBytes > 0) {
                    cudaOpts.add("gpu_mem_limit", Long.toString(gpuMemLimitBytes));
                }
                options.addCUDA(cudaOpts);
            }
            Objects.requireNonNull(configureOptions, "configureOptions").configure(options);
            return env.createSession(modelFile, options);
        } finally {
            // CUDA EP options 和 SessionOptions 在 createSession 后即完成使命，立即释放其原生资源。
            IOUtil.close(cudaOpts);
            IOUtil.close(options);
        }
    }

    static long resolveCudaGpuMemLimitBytes() {
        String value = System.getProperty(CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY);
        if (value == null || value.isBlank()) {
            return DEFAULT_CUDA_GPU_MEM_LIMIT_BYTES;
        }
        try {
            long parsed = Long.parseLong(value.trim());
            if (parsed >= 0) {
                return parsed;
            }
        } catch (NumberFormatException ignored) {
            // Fall back to the default (unlimited) for malformed configuration.
        }
        return DEFAULT_CUDA_GPU_MEM_LIMIT_BYTES;
    }

    @FunctionalInterface
    public interface SessionOptionsConfigurer {
        void configure(OrtSession.SessionOptions options) throws OrtException;
    }
}
