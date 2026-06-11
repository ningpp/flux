package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
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
import java.util.HashMap;

public class TextRecognitionModel extends BatchPredictor<PreProcessResult, List<RecognitionResult>> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, List<RecognitionResult>>> REGISTRY = new ModelRegistry<>();

    public static final Set<String> SUPPORT_MODELS = Set.of(
            "PP-OCRv6_medium_rec",
            "PP-OCRv6_small_rec",
            "PP-OCRv6_tiny_rec",
            "PP-OCRv5_server_rec",
            "PP-OCRv5_mobile_rec",
            "PP-OCRv4_server_rec",
            "PP-OCRv4_server_rec_doc",
            "PP-OCRv4_mobile_rec"
    );

    static {
        // v4/v5 recognition model factory
        ModelFactory<BatchPredictor<PreProcessResult, List<RecognitionResult>>> v4v5Factory =
                (modelDir, modelName, gpuIndex, env, customParams) -> {
                    List<ImageProcessor> preProcessors = List.of(
                            new OCRResizeNormImg()
                    );

                    PaddleRecognitionPredictor predictor = new PaddleRecognitionPredictor(
                            new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath(),
                            gpuIndex,
                            env,
                            preProcessors,
                            new CTCLabelDecode(new File(modelDir + File.separator + modelName, "config.yml").getAbsolutePath())
                    );
                    return new TextRecognitionPredictor(predictor);
                };

        // v6 recognition model factory (different file naming: inference.onnx, inference.yml)
        ModelFactory<BatchPredictor<PreProcessResult, List<RecognitionResult>>> v6Factory =
                (modelDir, modelName, gpuIndex, env, customParams) -> {
                    List<ImageProcessor> preProcessors = List.of(
                            new OCRResizeNormImg()
                    );

                    PaddleRecognitionPredictor predictor = new PaddleRecognitionPredictor(
                            new File(modelDir + File.separator + modelName + "_onnx", "inference.onnx").getAbsolutePath(),
                            gpuIndex,
                            env,
                            preProcessors,
                            new CTCLabelDecode(new File(modelDir + File.separator + modelName + "_onnx", "inference.yml").getAbsolutePath())
                    );
                    return new TextRecognitionPredictor(predictor);
                };

        REGISTRY.register(Set.of("PP-OCRv6_medium_rec", "PP-OCRv6_small_rec", "PP-OCRv6_tiny_rec"), v6Factory);
        REGISTRY.register(Set.of(
                "PP-OCRv5_server_rec",
                "PP-OCRv5_mobile_rec",
                "PP-OCRv4_server_rec",
                "PP-OCRv4_server_rec_doc",
                "PP-OCRv4_mobile_rec"
        ), v4v5Factory);
    }

    private final PaddleRecognitionPredictor predictor;

    public PaddleRecognitionPredictor getPredictor() {
        return predictor;
    }

    public TextRecognitionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex(), new HashMap<>());
    }

    public TextRecognitionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex) {
        this(modelDir, modelName, env, gpuIndex, new HashMap<>());
    }

    public TextRecognitionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex, Map<String, Object> customParams) {
        ModelFactory<BatchPredictor<PreProcessResult, List<RecognitionResult>>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("Not Supported Model: " + modelName));
        BatchPredictor<PreProcessResult, List<RecognitionResult>> temp = factory.create(modelDir, modelName, gpuIndex, env, customParams);
        if (temp instanceof TextRecognitionPredictor) {
            this.predictor = ((TextRecognitionPredictor) temp).getPredictor();
        } else {
            throw new FluxException("Unexpected predictor type");
        }
    }

    public static ModelRegistry<BatchPredictor<PreProcessResult, List<RecognitionResult>>> getRegistry() {
        return REGISTRY;
    }

    private static class TextRecognitionPredictor extends BatchPredictor<PreProcessResult, List<RecognitionResult>> {
        private final PaddleRecognitionPredictor predictor;

        TextRecognitionPredictor(PaddleRecognitionPredictor predictor) {
            this.predictor = predictor;
        }

        PaddleRecognitionPredictor getPredictor() {
            return predictor;
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
