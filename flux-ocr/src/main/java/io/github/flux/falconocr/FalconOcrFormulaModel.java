package io.github.flux.falconocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class FalconOcrFormulaModel extends BatchPredictor<PreProcessResult, TextResult> {

    private final FalconOcrModel model;

    public FalconOcrFormulaModel(final String modelRootDir,
                                 final String modelName,
                                 final int gpuIndex,
                                 final OrtEnvironment env,
                                 final Map<String, Object> customParams) {
        this.model = FalconOcrModel.getSharedInstance(modelRootDir, modelName, gpuIndex, env, customParams);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> images,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        return model.predictCategory(images, matManager, "formula");
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return model.processRgb(matManager, rgbMat, ndManager);
    }

    @Override
    public void close() throws Exception {
        model.close();
    }
}
