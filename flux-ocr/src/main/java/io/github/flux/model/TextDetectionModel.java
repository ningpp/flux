package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.predictor.PaddleDetectionPredictor;
import io.github.flux.paddle.processor.DBPostProcess;
import io.github.flux.paddle.processor.DetResize;
import io.github.flux.paddle.processor.LimitType;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TextDetectionModel extends BatchPredictor<PreProcessResult, TextDetectionResult> {

    public static final Set<String> SUPPORT_MODELS = Set.of(
            "PP-OCRv5_server_det",
            "PP-OCRv5_mobile_det",
            "PP-OCRv4_server_det",
            "PP-OCRv4_mobile_det"
    );

    private final PaddleDetectionPredictor predictor;

    public PaddleDetectionPredictor getPredictor() {
        return predictor;
    }

    public TextDetectionModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex());
    }

    public TextDetectionModel(String modelDir, String modelName, OrtEnvironment env, int gpuIndex) {
        if (!SUPPORT_MODELS.contains(modelName)) {
            throw new FluxException("Not Supported Model: " + modelName);
        }

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

        this.predictor = new PaddleDetectionPredictor(
                new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath(),
                gpuIndex,
                env,
                detResize,
                preProcessors,
                dbPostProcess
        );
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

    public List<TextDetectionResult> predict(List<String> images, MatManager matManager, NDManager manager,
                                             final Integer limitSideLen,
                                             final LimitType limitType,
                                             final Integer maxSideLimit,
                                             final Float thresh,
                                             final Float boxThresh,
                                             final Float unclipRatio) {
        List<TextDetectionResult> results = new ArrayList<>();
        for (String image : images) {
            results.add(predict(image, matManager, manager,
                    limitSideLen, limitType, maxSideLimit,
                    thresh, boxThresh, unclipRatio));
        }
        return results;
    }

    public TextDetectionResult predict(String image, MatManager matManager, NDManager manager,
                                       final Integer limitSideLen,
                                       final LimitType limitType,
                                       final Integer maxSideLimit,
                                       final Float thresh,
                                       final Float boxThresh,
                                       final Float unclipRatio) {
        Mat bgrImg = matManager.imread(image, Imgcodecs.IMREAD_COLOR_BGR);
        TextDetectionResult result = predict(bgrImg, matManager, manager,
                limitSideLen, limitType, maxSideLimit,
                thresh, boxThresh, unclipRatio);
        bgrImg.release();
        return result;
    }

    public TextDetectionResult predict(Mat image, MatManager matManager, NDManager manager,
                                       final Integer limitSideLen,
                                       final LimitType limitType,
                                       final Integer maxSideLimit,
                                       final Float thresh,
                                       final Float boxThresh,
                                       final Float unclipRatio) {
        return predictor.predict(image, matManager, manager, limitSideLen, limitType, maxSideLimit,
                thresh, boxThresh, unclipRatio);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }

}
