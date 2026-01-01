package io.github.flux.paddle.predictor;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.Crop;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.ResizeByShort;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PaddleDocOrientationPredictor extends BatchPredictor<PreProcessResult, ClassificationResult> {

    public static final Set<String> MODEL_NAMES = Set.of("PP-LCNet_x1_0_doc_ori");

    private final PaddleClassificationPredictor predictor;

    public PaddleDocOrientationPredictor(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex());
    }

    public PaddleDocOrientationPredictor(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("Not Supported Model: " + modelName);
        }
        List<String> labels = List.of("0", "90", "180", "270");
        List<ImageProcessor> preProcessors = List.of(
                new ResizeByShort(256, Image.Interpolation.BILINEAR),
                new Crop(224),
                new Normalize(
                        0.00392156862745098,
                        new double[]{0.485, 0.456, 0.406},
                        new double[]{0.229, 0.224, 0.225}
                ),
                new ToCHWImage()
        );

        String modelDir = modelRootDir + File.separator + modelName;
        this.predictor = new PaddleClassificationPredictor(
                new File(modelDir, "model.onnx").getAbsolutePath(),
                gpuIndex,
                env,
                preProcessors,
                labels
        );
    }

    @Override
    public List<ClassificationResult> doBatchPredict(List<PreProcessResult> mats, NDManager manager, Map<String, Object> extraParameters) {
        List<ClassificationResult> results = new ArrayList<>();
        for (PreProcessResult mat : mats) {
            results.add(predictor.predictProcessed(mat.mat(), 1).get(0));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return new PreProcessResult(predictor.process(rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

}
