// this code is convert from https://github.com/PaddlePaddle/PaddleX
// PaddleX's source code IS Licensed under the Apache License Version 2.0
/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.paddle.predictor;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.model.DocOrientationClassifyModel;
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

    static {
        DocOrientationClassifyModel.getRegistry().register(MODEL_NAMES, PaddleDocOrientationPredictor::new);
    }

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
    public List<ClassificationResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        List<ClassificationResult> results = new ArrayList<>();
        List<List<ClassificationResult>> batchKrs = predictor.doBatchPredict(mats, extraParameters);
        for (var krs : batchKrs) {
            results.add(krs.get(0));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(predictor.process(matManager, rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

}
