package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.RecognitionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.predictor.PaddleRecognitionPredictor;
import io.github.flux.paddle.processor.CTCLabelDecode;
import io.github.flux.paddle.processor.OCRResizeNormImg;
import org.opencv.core.Mat;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TextRecognitionModel extends BatchPredictor<PreProcessResult, List<RecognitionResult>> {

    public static final Set<String> SUPPORT_MODELS = Set.of(
            "PP-OCRv5_server_rec",
            "PP-OCRv5_mobile_rec",
            "PP-OCRv4_server_rec",
            "PP-OCRv4_server_rec_doc",
            "PP-OCRv4_mobile_rec"
    );

    private final PaddleRecognitionPredictor predictor;

    public PaddleRecognitionPredictor getPredictor() {
        return predictor;
    }

    public TextRecognitionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex());
    }

    public TextRecognitionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex) {
        if (!SUPPORT_MODELS.contains(modelName)) {
            throw new FluxException("Not Supported Model: " + modelName);
        }

        List<ImageProcessor> preProcessors = List.of(
                new OCRResizeNormImg()
        );

        this.predictor = new PaddleRecognitionPredictor(
                new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath(),
                gpuIndex,
                env,
                preProcessors,
                new CTCLabelDecode(new File(modelDir + File.separator + modelName, "config.yml").getAbsolutePath())
        );
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }

    @Override
    public List<List<RecognitionResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        try {
            return predictor.batchPredict(mats.stream().map(PreProcessResult::mat).toList(), matManager, manager);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(rgbMat, null);
    }
}
