package io.github.flux.util;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;

import java.util.Objects;

/**
 * 统一的 ONNX 会话创建工具。
 *
 * <p>集中处理几个系统性 GPU 内存问题（OCRPipelineGPUPerfTest 观测到的严重显存增长根因）：
 * <ol>
 *   <li>CUDA 会话默认开启 {@code memoryPatternOptimization}，会对<b>每种不同输入 shape</b>
 *       缓存一份内存分配方案，且该缓存在会话生命周期内不回收。OCR 每页 pad 后尺寸不同，
 *       导致 GPU 显存随页面尺寸种类无界累积。这里显式关闭它。</li>
 *   <li>CUDA EP 的 arena allocator 默认使用 {@code kNextPowerOfTwo} 策略，每次扩展时
 *       翻倍分配。OCR 输入尺寸多样（不同页面、不同裁剪区域），导致 arena 无界增长。
 *       这里改用 {@code kSameAsRequested}，按实际请求大小扩展。</li>
 *   <li>同一 OCRPipeline 会同时常驻 det/rec/layout/formula/table 等多个 CUDA session。
 *       默认无限制的 per-session CUDA arena 会让 native private bytes 接近多个单模型
 *       高水位的叠加。这里默认给每个 CUDA session 设置 4GiB arena 上限，可通过系统属性覆盖。</li>
 *   <li>OCR 输入和中间输出包含大量不同 shape 的 CPU tensor/native buffer。ORT CPU arena
 *       会在 session 生命周期内保留高水位，线上共享 Pipeline 不能靠关闭模型释放这些内存。
 *       这里默认关闭 CPU arena allocator，使请求内临时内存更及时归还给系统。</li>
 *   <li>{@link OrtSession.SessionOptions} 实现了 {@link AutoCloseable}，创建后必须
 *       {@code close()} 以释放原生 {@code OrtSessionOptions} 结构体，否则造成原生内存泄露。
 *       这里在 {@code createSession} 之后立即关闭。</li>
 * </ol>
 */
public final class OnnxSessionUtil {

    static final String CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY = "flux.ocr.onnx.cudaGpuMemLimitBytes";
    static final String CPU_ARENA_ALLOCATOR_ENABLED_PROPERTY = "flux.ocr.onnx.cpuArenaAllocatorEnabled";
    private static final long DEFAULT_CUDA_GPU_MEM_LIMIT_BYTES = 4L * 1024L * 1024L * 1024L;

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
                // Arena 按实际请求大小扩展，而非翻倍，防止 GPU 显存随输入尺寸种类无界增长。
                cudaOpts.add("arena_extend_strategy", "kSameAsRequested");
                long gpuMemLimitBytes = resolveCudaGpuMemLimitBytes();
                if (gpuMemLimitBytes > 0) {
                    cudaOpts.add("gpu_mem_limit", Long.toString(gpuMemLimitBytes));
                }
                options.addCUDA(cudaOpts);
            }
            // 关闭内存模式优化，避免按输入 shape 缓存分配方案导致 GPU 显存无界增长。
            options.setMemoryPatternOptimization(false);
            // 默认关闭 CPU arena，避免共享 session 在请求后长期保留 OCR 大 shape 的 CPU 高水位。
            options.setCPUArenaAllocator(resolveCpuArenaAllocatorEnabled());
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
            // Fall back to the conservative default for malformed configuration.
        }
        return DEFAULT_CUDA_GPU_MEM_LIMIT_BYTES;
    }

    static boolean resolveCpuArenaAllocatorEnabled() {
        return Boolean.parseBoolean(System.getProperty(CPU_ARENA_ALLOCATOR_ENABLED_PROPERTY, "false"));
    }

    @FunctionalInterface
    public interface SessionOptionsConfigurer {
        void configure(OrtSession.SessionOptions options) throws OrtException;
    }
}
