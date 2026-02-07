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
import java.util.HashMap;

public class FormulaRecognitionModel extends BatchPredictor<PreProcessResult, TextResult> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, TextResult>> REGISTRY = new ModelRegistry<>();

    static {
        // Trigger class loading to ensure models register themselves
        try {
            Class.forName(ByteDanceDolphinElementModel.class.getName());
            Class.forName(NougatLatexFormulaModel.class.getName());
            Class.forName(PaddleFormulaRecognitionPredictor.class.getName());
            Class.forName(Pix2TextFormulaRecognitionPredictor.class.getName());
            Class.forName(TexTellerPredictor.class.getName());
            Class.forName(UnirecPredictor.class.getName());
        } catch (ClassNotFoundException e) {
            throw new FluxException("Failed to load model classes", e);
        }

        // GraniteDoclingFormulaModel needs special handling due to extra maxLength parameter
        REGISTRY.register(GraniteDoclingFormulaModel.MODEL_NAMES,
                (dir, name, gpu, env, customParams) -> new GraniteDoclingFormulaModel(dir, name, gpu, env, 8192, customParams));
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
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env(), new HashMap<>());
    }

    public FormulaRecognitionModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env) {
        this(modelRootDir, modelName, gpuIndex, env, new HashMap<>());
    }

    public FormulaRecognitionModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env,
                                   final Map<String, Object> customParams) {
        ModelFactory<BatchPredictor<PreProcessResult, TextResult>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("not supported formula model: " + modelName));
        this.predictor = factory.create(modelRootDir, modelName, gpuIndex, env, customParams);
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
