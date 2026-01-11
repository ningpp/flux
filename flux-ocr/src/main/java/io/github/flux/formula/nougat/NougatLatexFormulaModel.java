package io.github.flux.formula.nougat;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class NougatLatexFormulaModel extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "nougat-latex-base"
    );

    private final NougatLatexEncoderModel encoderModel;
    private final NougatLatexDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final NougatImageProcessor preProcessor;

    public NougatLatexFormulaModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported nougat latex model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.preProcessor = new NougatImageProcessor();
            this.encoderModel = new NougatLatexEncoderModel(new File(modelDir, "encoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, preProcessor);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new NougatLatexDecoderModel(new File(modelDir, "decoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096, 1, 2, 0, tokenizer);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        NDArray ndArray = preProcessor.process(rgbMat, manager);
        return new PreProcessResult(null, ndArray);
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> batch, NDManager manager, Map<String, Object> extraParameters) {
        try {
            float[][][] encoderResults = encoderModel.batchPredict(PreProcessResult.getNDArrays(batch), manager);
            return decoderModel.batchPredict(encoderResults, manager);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(tokenizer);
    }
}
