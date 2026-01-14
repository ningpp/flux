package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.dolphin.ByteDanceDolphinElementModel;
import io.github.flux.dolphin.ByteDanceDolphinFormulaModel;
import io.github.flux.exception.FluxException;
import io.github.flux.formula.nougat.NougatLatexFormulaModel;
import io.github.flux.formula.paddle.PaddleFormulaRecognitionPredictor;
import io.github.flux.formula.pix2text.Pix2TextFormulaRecognitionPredictor;
import io.github.flux.unirec.UnirecFormulaModel;
import io.github.flux.unirec.UnirecPredictor;
import io.github.flux.util.CollectionUtil;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class FormulaRecognitionModel extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    private final BatchPredictor<PreProcessResult, FormulaRecognitionResult> predictor;

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            ByteDanceDolphinElementModel.MODEL_NAMES,
            NougatLatexFormulaModel.MODEL_NAMES,
            PaddleFormulaRecognitionPredictor.MODEL_NAMES,
            Pix2TextFormulaRecognitionPredictor.MODEL_NAMES,
            UnirecPredictor.MODEL_NAMES
    ));

    public FormulaRecognitionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env());
    }

    public FormulaRecognitionModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env) {
        if (ByteDanceDolphinElementModel.MODEL_NAMES.contains(modelName)) {
            this.predictor = new ByteDanceDolphinFormulaModel(modelRootDir, modelName, gpuIndex, env, OnnxJavaType.FLOAT);
        } else if (NougatLatexFormulaModel.MODEL_NAMES.contains(modelName)) {
            this.predictor = new NougatLatexFormulaModel(modelRootDir, modelName, gpuIndex, env);
        } else if (PaddleFormulaRecognitionPredictor.MODEL_NAMES.contains(modelName)) {
            this.predictor = new PaddleFormulaRecognitionPredictor(modelRootDir, modelName, gpuIndex, env);
        } else if (Pix2TextFormulaRecognitionPredictor.MODEL_NAMES.contains(modelName)) {
            this.predictor = new Pix2TextFormulaRecognitionPredictor(modelRootDir, modelName, gpuIndex, env);
        } else if (UnirecPredictor.MODEL_NAMES.contains(modelName)) {
            this.predictor = new UnirecFormulaModel(new UnirecPredictor(modelRootDir, modelName, gpuIndex, env));
        } else {
            throw new FluxException("not supported formula model: " + modelName);
        }
    }


    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> pprs, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(pprs, matManager, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

}
