package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.predictor.PaddleDocOrientationPredictor;
import io.github.flux.util.CollectionUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashMap;

public class DocOrientationClassifyModel extends BatchPredictor<PreProcessResult, ClassificationResult> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, ClassificationResult>> REGISTRY = new ModelRegistry<>();

    static {
        // Trigger class loading to ensure models register themselves
        try {
            Class.forName(PaddleDocOrientationPredictor.class.getName());
        } catch (ClassNotFoundException e) {
            throw new FluxException("Failed to load model classes", e);
        }
    }

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            PaddleDocOrientationPredictor.MODEL_NAMES
    ));

    @Override
    public void close() throws Exception {
        predictor.close();
    }

    private final BatchPredictor<PreProcessResult, ClassificationResult> predictor;

    public DocOrientationClassifyModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex(), new HashMap<>());
    }

    public DocOrientationClassifyModel(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex) {
        this(modelRootDir, modelName, env, gpuIndex, new HashMap<>());
    }

    public DocOrientationClassifyModel(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex, Map<String, Object> customParams) {
        ModelFactory<BatchPredictor<PreProcessResult, ClassificationResult>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("not supported doc orientation model: " + modelName));
        predictor = factory.create(modelRootDir, modelName, gpuIndex, env, customParams);
    }

    public static ModelRegistry<BatchPredictor<PreProcessResult, ClassificationResult>> getRegistry() {
        return REGISTRY;
    }

    @Override
    public List<ClassificationResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, matManager, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

}
