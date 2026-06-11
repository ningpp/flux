package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.predictor.PaddleDetectionPredictor;
import io.github.flux.paddle.processor.DBPostProcess;
import io.github.flux.paddle.processor.DetResize;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.LimitType;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashMap;

public class TextDetectionModel extends BatchPredictor<PreProcessResult, TextDetectionResult> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, TextDetectionResult>> REGISTRY = new ModelRegistry<>();

    public static final Set<String> SUPPORT_MODELS = Set.of(
            "PP-OCRv6_medium_det",
            "PP-OCRv6_small_det",
            "PP-OCRv6_tiny_det",
            "PP-OCRv5_server_det",
            "PP-OCRv5_mobile_det",
            "PP-OCRv4_server_det",
            "PP-OCRv4_mobile_det"
    );

    static {
        // v4/v5 detection model factory
        ModelFactory<BatchPredictor<PreProcessResult, TextDetectionResult>> v4v5Factory =
                (modelDir, modelName, gpuIndex, env, customParams) -> {
                    DetResize detResize = new DetResize(0, 960, LimitType.MAX);
                    List<ImageProcessor> preProcessors = List.of(
                            new Normalize(
                                    1.0 / 255.0,
                                    new double[]{0.485, 0.456, 0.406},
                                    new double[]{0.229, 0.224, 0.225}
                            ),
                            new ToCHWImage()
                    );
                    final DBPostProcess dbPostProcess = new DBPostProcess(
                            0.3f, 0.6f, 1.5f, 1000, "fast", "quad"
                    );

                    PaddleDetectionPredictor predictor = new PaddleDetectionPredictor(
                            new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath(),
                            gpuIndex,
                            env,
                            detResize,
                            preProcessors,
                            dbPostProcess
                    );
                    return new TextDetectionPredictor(predictor);
                };

        // v6 detection model factory (different resize params, file naming, and maxCandidates)
        ModelFactory<BatchPredictor<PreProcessResult, TextDetectionResult>> v6Factory =
                (modelDir, modelName, gpuIndex, env, customParams) -> {
                    DetResize detResize = new DetResize(0, 64, LimitType.MIN, null, 4000, null, null);
                    List<ImageProcessor> preProcessors = List.of(
                            new Normalize(
                                    1.0 / 255.0,
                                    new double[]{0.485, 0.456, 0.406},
                                    new double[]{0.229, 0.224, 0.225}
                            ),
                            new ToCHWImage()
                    );
                    final DBPostProcess dbPostProcess = new DBPostProcess(
                            0.3f, 0.6f, 1.5f, 3000, "fast", "quad"
                    );

                    PaddleDetectionPredictor predictor = new PaddleDetectionPredictor(
                            new File(modelDir + File.separator + modelName + "_onnx", "inference.onnx").getAbsolutePath(),
                            gpuIndex,
                            env,
                            detResize,
                            preProcessors,
                            dbPostProcess
                    );
                    return new TextDetectionPredictor(predictor);
                };

        REGISTRY.register(Set.of("PP-OCRv6_medium_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det"), v6Factory);
        REGISTRY.register(Set.of(
                "PP-OCRv5_server_det",
                "PP-OCRv5_mobile_det",
                "PP-OCRv4_server_det",
                "PP-OCRv4_mobile_det"
        ), v4v5Factory);
    }

    private final PaddleDetectionPredictor predictor;

    public PaddleDetectionPredictor getPredictor() {
        return predictor;
    }

    public TextDetectionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex(), new HashMap<>());
    }

    public TextDetectionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex) {
        this(modelDir, modelName, env, gpuIndex, new HashMap<>());
    }

    public TextDetectionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex, Map<String, Object> customParams) {
        ModelFactory<BatchPredictor<PreProcessResult, TextDetectionResult>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("Not Supported Model: " + modelName));
        BatchPredictor<PreProcessResult, TextDetectionResult> temp = factory.create(modelDir, modelName, gpuIndex, env, customParams);
        if (temp instanceof TextDetectionPredictor) {
            this.predictor = ((TextDetectionPredictor) temp).getPredictor();
        } else {
            throw new FluxException("Unexpected predictor type");
        }
    }

    public static ModelRegistry<BatchPredictor<PreProcessResult, TextDetectionResult>> getRegistry() {
        return REGISTRY;
    }

    private static class TextDetectionPredictor extends BatchPredictor<PreProcessResult, TextDetectionResult> {
        private final PaddleDetectionPredictor predictor;

        TextDetectionPredictor(PaddleDetectionPredictor predictor) {
            this.predictor = predictor;
        }

        PaddleDetectionPredictor getPredictor() {
            return predictor;
        }

        @Override
        public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager,
                                                        Map<String, Object> extraParameters) {
            try {
                return predictor.batchPredict(PreProcessResult.getMats(mats), matManager, manager,
                        ParameterUtil.getInteger(extraParameters, "det.limitSideLen"),
                        ParameterUtil.getLimitType(extraParameters, "det.limitType"),
                        ParameterUtil.getInteger(extraParameters, "det.maxSideLimit"),
                        ParameterUtil.getFloat(extraParameters, "det.thresh"),
                        ParameterUtil.getFloat(extraParameters, "det.boxThresh"),
                        ParameterUtil.getFloat(extraParameters, "det.unclipRatio"));
            } catch (Exception e) {
                throw new FluxException(e);
            }
        }

        @Override
        public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
            return new PreProcessResult(rgbMat, null);
        }

        @Override
        public void close() throws Exception {
            predictor.close();
        }
    }

    @Override
    public List<TextDetectionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager,
                                                    Map<String, Object> extraParameters) {
        try {
            return predictor.batchPredict(PreProcessResult.getMats(mats), matManager, manager,
                        ParameterUtil.getInteger(extraParameters, "det.limitSideLen"),
                        ParameterUtil.getLimitType(extraParameters, "det.limitType"),
                        ParameterUtil.getInteger(extraParameters, "det.maxSideLimit"),
                        ParameterUtil.getFloat(extraParameters, "det.thresh"),
                        ParameterUtil.getFloat(extraParameters, "det.boxThresh"),
                        ParameterUtil.getFloat(extraParameters, "det.unclipRatio"));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(rgbMat, null);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }

}
