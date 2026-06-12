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
package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelFactory;
import io.github.flux.core.ModelParam;
import io.github.flux.core.ModelRegistry;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.predictor.PaddleTextLineOrientationPredictor;
import io.github.flux.util.CollectionUtil;
import org.opencv.core.Mat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Text line orientation classification model facade.
 * Supports PP-LCNet_x1_0_textline_ori and PP-LCNet_x0_25_textline_ori models.
 */
public class TextLineOrientationModel extends BatchPredictor<PreProcessResult, ClassificationResult> {

    private static final ModelRegistry<BatchPredictor<PreProcessResult, ClassificationResult>> REGISTRY = new ModelRegistry<>();

    static {
        try {
            Class.forName(PaddleTextLineOrientationPredictor.class.getName());
        } catch (ClassNotFoundException e) {
            throw new FluxException("Failed to load textline orientation model classes", e);
        }
    }

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            PaddleTextLineOrientationPredictor.MODEL_NAMES
    ));

    @Override
    public void close() throws Exception {
        predictor.close();
    }

    private final BatchPredictor<PreProcessResult, ClassificationResult> predictor;

    public TextLineOrientationModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.env(), param.gpuIndex(), new HashMap<>());
    }

    public TextLineOrientationModel(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex) {
        this(modelRootDir, modelName, env, gpuIndex, new HashMap<>());
    }

    public TextLineOrientationModel(String modelRootDir, String modelName, OrtEnvironment env, int gpuIndex, Map<String, Object> customParams) {
        ModelFactory<BatchPredictor<PreProcessResult, ClassificationResult>> factory = REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("not supported textline orientation model: " + modelName));
        predictor = factory.create(modelRootDir, modelName, gpuIndex, env, customParams);
    }

    public static ModelRegistry<BatchPredictor<PreProcessResult, ClassificationResult>> getRegistry() {
        return REGISTRY;
    }

    @Override
    public List<ClassificationResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, matManager, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

}
