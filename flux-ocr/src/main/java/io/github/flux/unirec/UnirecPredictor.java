// this code is convert from https://github.com/Topdu/OpenOCR
// OpenOCR's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import io.github.flux.model.FormulaRecognitionModel;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class UnirecPredictor extends BatchPredictor<PreProcessResult, TextResult> {

    public static final List<String> MODEL_NAMES = List.of(
            "unirec-0.1b"
    );

    // Shared instance cache to avoid creating multiple expensive model instances
    private static final Map<InstanceKey, UnirecPredictor> INSTANCE_CACHE = new ConcurrentHashMap<>();

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, UnirecFormulaModel::new);
    }

    /**
     * Key for caching shared instances based on model configuration.
     */
    record InstanceKey(String modelRootDir, String modelName, int gpuIndex) {
    }

    private final UnirecEncoderModel encoderModel;
    private final UnirecDecoderModel decoderModel;

    /**
     * Gets a shared instance of UnirecPredictor for the given configuration.
     * If an instance with the same configuration already exists, it will be reused.
     * This is important because the model is expensive to create (loads ONNX models).
     *
     * @param modelRootDir the root directory containing model files
     * @param modelName the name of the model
     * @param gpuIndex the GPU index to use (-1 for CPU)
     * @param env the ONNX runtime environment
     * @return a shared instance of the model
     */
    public static UnirecPredictor getSharedInstance(final String modelRootDir,
                                                     final String modelName,
                                                     final int gpuIndex,
                                                     final OrtEnvironment env) {
        InstanceKey key = new InstanceKey(modelRootDir, modelName, gpuIndex);
        return INSTANCE_CACHE.computeIfAbsent(key, k ->
            new UnirecPredictor(modelRootDir, modelName, gpuIndex, env));
    }

    /**
     * Private constructor - use getSharedInstance() to obtain instances.
     */
    private UnirecPredictor(final String modelRootDir,
                            final String modelName,
                            final int gpuIndex,
                            final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported unirec model: " + modelName);
        }
        this.encoderModel = new UnirecEncoderModel(
                new File(modelRootDir + File.separator + modelName, "unirec_encoder.onnx").getAbsolutePath(), gpuIndex, env);
        this.decoderModel = new UnirecDecoderModel(
                new File(modelRootDir + File.separator + modelName, "unirec_decoder.onnx").getAbsolutePath(),
                new File(modelRootDir + File.separator + modelName, "unirec_tokenizer_mapping.json").getAbsolutePath(),
                gpuIndex, env);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        List<TextResult> results = new ArrayList<>(mats.size());
        for (PreProcessResult ppr : mats) {
            UnirecEncoderModelPredictResult encodeResult = encoderModel.predict(ppr, matManager, manager);
            results.add(decoderModel.predict(encodeResult));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(new UnirecProcessor().process(matManager, rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
    }
}
