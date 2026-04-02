package io.github.flux.llamajcpp;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelInstanceKey;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.model.FormulaRecognitionModel;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class LlamaJCppOcrModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of("LlamaJCpp-OCR");

    private static final Map<ModelInstanceKey, LlamaJCppOcrModel> INSTANCE_CACHE = new ConcurrentHashMap<>();

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, LlamaJCppOcrModel::getSharedInstance);
    }

    private final ModelInstanceKey cacheKey;
    private final LlamaJCppOcrSession session;

    public static LlamaJCppOcrModel getSharedInstance(final String modelRootDir,
                                                      final String modelName,
                                                      final int gpuIndex,
                                                      final OrtEnvironment env,
                                                      final Map<String, Object> customParams) {
        final Map<String, Object> normalized = normalizeCustomParams(customParams);
        final ModelInstanceKey key = new ModelInstanceKey(modelRootDir, modelName, gpuIndex, normalized);
        return INSTANCE_CACHE.computeIfAbsent(key, ignored ->
                new LlamaJCppOcrModel(modelRootDir, modelName, gpuIndex, normalized, key));
    }

    private LlamaJCppOcrModel(final String modelRootDir,
                              final String modelName,
                              final int gpuIndex,
                              final Map<String, Object> customParams,
                              final ModelInstanceKey cacheKey) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported llamaj.cpp model: " + modelName);
        }
        this.cacheKey = cacheKey;
        this.session = new LlamaJCppOcrSession(LlamaJCppConfig.from(modelRootDir, modelName, gpuIndex, customParams), gpuIndex);
    }

    @Override
    public List<TextResult> doBatchPredict(final List<PreProcessResult> pprs,
                                           final MatManager matManager,
                                           final NDManager ndManager,
                                           final Map<String, Object> extraParameters) {
        final Map<String, Object> params = extraParameters == null ? Map.of() : extraParameters;
        final String prompt = String.valueOf(params.getOrDefault("prompt", "Extract all text from the image."));
        final List<TextResult> results = new ArrayList<>(pprs.size());
        for (PreProcessResult ppr : pprs) {
            if (ppr == null || ppr.mat() == null || ppr.mat().empty()) {
                throw new FluxException("llamaj.cpp OCR requires a non-empty RGB Mat input");
            }
            results.add(session.predict(ppr.mat(), prompt));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(final MatManager matManager, final Mat rgbMat, final NDManager ndManager) {
        return new PreProcessResult(rgbMat, null);
    }

    @Override
    public void close() {
        INSTANCE_CACHE.remove(cacheKey);
        session.close();
    }

    private static Map<String, Object> normalizeCustomParams(final Map<String, Object> customParams) {
        return customParams == null || customParams.isEmpty() ? Map.of() : Map.copyOf(customParams);
    }
}
