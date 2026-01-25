package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.paddle.predictor.PaddleDocOrientationPredictor;
import io.github.flux.util.CollectionUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class DocOrientationClassifyModel extends BatchPredictor<PreProcessResult, ClassificationResult> {

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            PaddleDocOrientationPredictor.MODEL_NAMES
    ));

    @Override
    public void close() throws Exception {
        predictor.close();
    }

    private final BatchPredictor<PreProcessResult, ClassificationResult> predictor;

    public DocOrientationClassifyModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex());
    }

    public DocOrientationClassifyModel(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex) {
        predictor = new PaddleDocOrientationPredictor(modelRootDir, modelName, env, gpuIndex);
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
