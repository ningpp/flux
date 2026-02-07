package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.docling.DoclingLayoutModel;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.PaddleLayoutModel;
import io.github.flux.util.CollectionUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class LayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    private static final ModelRegistry<BatchPredictor<ProcessedMat, List<ObjectDetectionResult>>> REGISTRY = new ModelRegistry<>();

    static {
        // Trigger class loading to ensure models register themselves
        try {
            Class.forName(DoclingLayoutModel.class.getName());
            Class.forName(PaddleLayoutModel.class.getName());
        } catch (ClassNotFoundException e) {
            throw new FluxException("Failed to load model classes", e);
        }
    }

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            DoclingLayoutModel.MODEL_NAMES,
            PaddleLayoutModel.MODEL_NAMES
    ));

    private final BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> predictor;

    public LayoutModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env());
    }

    public LayoutModel(final String modelRootDir,
                       final String modelName,
                       final int gpuIndex,
                       final OrtEnvironment env) {
        ModelFactory<BatchPredictor<ProcessedMat, List<ObjectDetectionResult>>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("not supported layout model: " + modelName));
        predictor = factory.create(modelRootDir, modelName, gpuIndex, env);
    }

    public static ModelRegistry<BatchPredictor<ProcessedMat, List<ObjectDetectionResult>>> getRegistry() {
        return REGISTRY;
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, matManager, manager, extraParameters);
    }

    @Override
    public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }
}
