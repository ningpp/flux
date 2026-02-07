package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.TextResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
import io.github.flux.core.PreProcessResult;
import io.github.flux.dolphin.ByteDanceDolphinElementModel;
import io.github.flux.dolphin.ByteDanceDolphinFormulaModel;
import io.github.flux.exception.FluxException;
import io.github.flux.formula.granite.GraniteDoclingFormulaModel;
import io.github.flux.formula.nougat.NougatLatexFormulaModel;
import io.github.flux.formula.paddle.PaddleFormulaRecognitionPredictor;
import io.github.flux.formula.pix2text.Pix2TextFormulaRecognitionPredictor;
import io.github.flux.formula.texteller.TexTellerPredictor;
import io.github.flux.unirec.UnirecFormulaModel;
import io.github.flux.unirec.UnirecPredictor;
import io.github.flux.util.CollectionUtil;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class FormulaRecognitionModel extends BatchPredictor<PreProcessResult, TextResult> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, TextResult>> REGISTRY = new ModelRegistry<>();

    static {
        REGISTRY.register(ByteDanceDolphinElementModel.MODEL_NAMES, ByteDanceDolphinFormulaModel::new);
        REGISTRY.register(GraniteDoclingFormulaModel.MODEL_NAMES,
                (dir, name, gpu, env) -> new GraniteDoclingFormulaModel(dir, name, gpu, env, 8192));
        REGISTRY.register(NougatLatexFormulaModel.MODEL_NAMES, NougatLatexFormulaModel::new);
        REGISTRY.register(PaddleFormulaRecognitionPredictor.MODEL_NAMES, PaddleFormulaRecognitionPredictor::new);
        REGISTRY.register(Pix2TextFormulaRecognitionPredictor.MODEL_NAMES, Pix2TextFormulaRecognitionPredictor::new);
        REGISTRY.register(TexTellerPredictor.MODEL_NAMES, TexTellerPredictor::new);
        REGISTRY.register(UnirecPredictor.MODEL_NAMES,
                (dir, name, gpu, env) -> new UnirecFormulaModel(new UnirecPredictor(dir, name, gpu, env)));
    }

    private final BatchPredictor<PreProcessResult, TextResult> predictor;

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            ByteDanceDolphinElementModel.MODEL_NAMES,
            GraniteDoclingFormulaModel.MODEL_NAMES,
            NougatLatexFormulaModel.MODEL_NAMES,
            PaddleFormulaRecognitionPredictor.MODEL_NAMES,
            Pix2TextFormulaRecognitionPredictor.MODEL_NAMES,
            TexTellerPredictor.MODEL_NAMES,
            UnirecPredictor.MODEL_NAMES
    ));

    public FormulaRecognitionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env());
    }

    public FormulaRecognitionModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env) {
        ModelFactory<BatchPredictor<PreProcessResult, TextResult>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("not supported formula model: " + modelName));
        this.predictor = factory.create(modelRootDir, modelName, gpuIndex, env);
    }

    public static ModelRegistry<BatchPredictor<PreProcessResult, TextResult>> getRegistry() {
        return REGISTRY;
    }


    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> pprs, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(pprs, matManager, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

}
