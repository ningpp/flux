package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ModelParam;
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
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported formula model: " + modelName);
        }
        if (DoclingLayoutModel.MODEL_NAMES.contains(modelName)) {
            predictor = new DoclingLayoutModel(modelRootDir, modelName, gpuIndex, env);
        } else {
            predictor = new PaddleLayoutModel(modelRootDir, modelName, gpuIndex, env);
        }
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, manager, extraParameters);
    }

    @Override
    public ProcessedMat processRgb(Mat rgbMat, NDManager manager) {
        return predictor.processRgb(rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }
}
