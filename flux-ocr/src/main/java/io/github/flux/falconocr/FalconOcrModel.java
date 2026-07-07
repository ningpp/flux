package io.github.flux.falconocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelInstanceKey;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class FalconOcrModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of("Falcon-OCR", "Falcon-OCR-ONNX");

    private static final int DEFAULT_MAX_SEQ_LEN = 8192;

    /**
     * 共享实例缓存。
     *
     * <p>修复点：旧实现使用无界 {@code ConcurrentHashMap} 且只增不减，每个不同的
     * {@link ModelInstanceKey}（含 modelRootDir/modelName/gpuIndex/customParams）都会创建并永久驻留一个
     * {@link FalconOcrModel}。该实例持有 HuggingFaceTokenizer、ONNX OrtSession（GPU 下占显存）与
     * OrtIoBindingNative（CUDA 资源），长期运行会持续累积直至 OOM，且在 GPU 上永久占用显存；同时
     * {@code close()} 直接销毁共享实例却不从缓存移除，造成悬挂引用与 use-after-close。
     *
     * <p>现改为引用计数：仅当引用计数归零时才真正关闭底层资源并从缓存中移除，既保留“模型复用”的性能优势，
     * 又避免无界增长与“共享实例被其中一个持有者关闭后其他持有者 use-after-close”的问题。
     */
    private static final Map<ModelInstanceKey, CacheEntry> INSTANCE_CACHE = new ConcurrentHashMap<>();
    private static final Object CACHE_LOCK = new Object();

    /**
     * 缓存条目：持有共享模型实例及其引用计数。
     */
    private static final class CacheEntry {
        final FalconOcrModel model;
        int refCount;

        CacheEntry(FalconOcrModel model) {
            this.model = model;
            this.refCount = 1;
        }
    }

    public static FalconOcrModel getSharedInstance(final String modelRootDir,
                                                   final String modelName,
                                                   final int gpuIndex,
                                                   final OrtEnvironment env,
                                                   final Map<String, Object> customParams) {
        ModelInstanceKey key = new ModelInstanceKey(modelRootDir, modelName, gpuIndex, customParams);
        synchronized (CACHE_LOCK) {
            CacheEntry entry = INSTANCE_CACHE.get(key);
            if (entry != null) {
                entry.refCount++;
                return entry.model;
            }
            FalconOcrModel created = new FalconOcrModel(modelRootDir, modelName, gpuIndex, env, customParams);
            INSTANCE_CACHE.put(key, new CacheEntry(created));
            return created;
        }
    }

    /**
     * 当前缓存中的共享实例数量，主要用于内存泄露验证（归零表示无悬挂引用）。
     */
    public static int sharedInstanceCount() {
        return INSTANCE_CACHE.size();
    }

    /**
     * 关闭并清空所有共享实例（兜底/测试用）。
     */
    public static void clearSharedInstances() {
        synchronized (CACHE_LOCK) {
            for (CacheEntry entry : INSTANCE_CACHE.values()) {
                entry.model.doClose();
            }
            INSTANCE_CACHE.clear();
        }
    }

    private final ModelInstanceKey key;
    private final HuggingFaceTokenizer tokenizer;
    private final FalconOcrTokenKvModel tokenKvModel;
    private volatile boolean closed = false;

    private FalconOcrModel(final String modelRootDir,
                           final String modelName,
                           final int gpuIndex,
                           final OrtEnvironment env,
                           final Map<String, Object> customParams) {
        this.key = new ModelInstanceKey(modelRootDir, modelName, gpuIndex, customParams);
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported Falcon-OCR model: " + modelName);
        }
        final String modelDir = modelRootDir + File.separator + modelName;
        Integer maxNewTokens = getOptionalPositiveInt(customParams, "maxNewTokens");
        int maxSeqLen = readMaxSeqLen(modelDir);
        try {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.tokenKvModel = new FalconOcrTokenKvModel(
                    new File(modelDir, "falcon_ocr_kv_token.onnx").getAbsolutePath(),
                    gpuIndex,
                    env,
                    maxNewTokens,
                    maxSeqLen
            );
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    List<TextResult> predictCategory(List<PreProcessResult> images,
                                     MatManager matManager,
                                     String category) {
        try {
            List<FalconOcrProcessor.Preprocessed> items = new ArrayList<>(images.size());
            for (PreProcessResult image : images) {
                items.add(FalconOcrProcessor.process(matManager, image.mat(), tokenizer, category));
            }
            FalconOcrProcessor.BatchPreprocessed batch = FalconOcrProcessor.batchPad(items);
            long[][] generatedIds = tokenKvModel.predict(batch);
            List<TextResult> results = new ArrayList<>(generatedIds.length);
            for (long[] ids : generatedIds) {
                results.add(new TextResult(decodeGenerated(ids), ids, -1));
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> images,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        String category = String.valueOf(
                extraParameters == null ? "formula" : extraParameters.getOrDefault("category", "formula"));
        return predictCategory(images, matManager, category);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(rgbMat, null);
    }

    private String decodeGenerated(long[] ids) {
        return tokenizer.decode(ids, false)
                .replace("<|end_of_query|>", "")
                .replace("<|end_of_text|>", "")
                .strip();
    }

    private static Integer getOptionalPositiveInt(Map<String, Object> customParams, String key) {
        if (customParams == null || !customParams.containsKey(key)) {
            return null;
        }
        Object value = customParams.get(key);
        int parsed;
        if (value instanceof Number number) {
            parsed = number.intValue();
        } else {
            parsed = Integer.parseInt(String.valueOf(value));
        }
        if (parsed <= 0) {
            throw new FluxException("Falcon-OCR " + key + " must be positive, got: " + parsed);
        }
        return parsed;
    }

    private static int readMaxSeqLen(String modelDir) {
        File configFile = new File(modelDir, "config.json");
        try {
            JsonObject config = JsonParser.parseString(
                    Files.readString(configFile.toPath(), StandardCharsets.UTF_8)).getAsJsonObject();
            if (config.has("max_seq_len")) {
                return config.get("max_seq_len").getAsInt();
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
        return DEFAULT_MAX_SEQ_LEN;
    }

    @Override
    public void close() {
        synchronized (CACHE_LOCK) {
            if (closed) {
                return;
            }
            CacheEntry entry = INSTANCE_CACHE.get(key);
            if (entry == null || entry.model != this) {
                // 未命中（理论上不应发生）——直接关闭，避免资源悬挂泄漏。
                closed = true;
                doClose();
                return;
            }
            if (--entry.refCount <= 0) {
                INSTANCE_CACHE.remove(key);
                closed = true;
                doClose();
            }
        }
    }

    private void doClose() {
        IOUtil.close(tokenKvModel);
        IOUtil.close(tokenizer);
    }
}
