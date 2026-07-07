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
import io.github.flux.core.ModelInstanceKey;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class UnirecPredictor extends BatchPredictor<PreProcessResult, TextResult> {

    public static final List<String> MODEL_NAMES = List.of(
            "unirec-0.1b"
    );

    // Shared instance cache to avoid creating multiple expensive model instances
    private static final Map<ModelInstanceKey, UnirecPredictor> INSTANCE_CACHE = new ConcurrentHashMap<>();

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, UnirecFormulaModel::new);
    }

    private final ModelInstanceKey key;
    private final UnirecEncoderModel encoderModel;
    private final UnirecDecoderModel decoderModel;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicInteger refCount = new AtomicInteger();

    /**
     * Gets a shared instance of UnirecPredictor for the given configuration.
     * If an instance with the same configuration already exists, it will be reused.
     * This is important because the model is expensive to create (loads ONNX models).
     *
     * @param modelRootDir the root directory containing model files
     * @param modelName the name of the model
     * @param gpuIndex the GPU index to use (-1 for CPU)
     * @param env the ONNX runtime environment
     * @param customParams custom initialization parameters (e.g., encoder GPU, decoder GPU)
     * @return a shared instance of the model
     */
    public static UnirecPredictor getSharedInstance(final String modelRootDir,
                                                     final String modelName,
                                                     final int gpuIndex,
                                                     final OrtEnvironment env,
                                                     final Map<String, Object> customParams) {
        ModelInstanceKey key = new ModelInstanceKey(modelRootDir, modelName, gpuIndex, customParams);
        UnirecPredictor predictor = INSTANCE_CACHE.compute(key, (k, existing) -> {
            if (existing == null || existing.closed.get()) {
                existing = new UnirecPredictor(k, modelRootDir, modelName, gpuIndex, env, customParams);
            }
            existing.retain();
            return existing;
        });
        return predictor;
    }

    private UnirecPredictor(ModelInstanceKey key, final String modelRootDir,
                            final String modelName,
                            final int gpuIndex,
                            final OrtEnvironment env,
                            final Map<String, Object> customParams) {
        this.key = key;
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

    private void retain() {
        refCount.incrementAndGet();
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        List<TextResult> results = new ArrayList<>(mats.size());
        for (PreProcessResult ppr : mats) {
            UnirecEncoderModelPredictResult encodeResult = encoderModel.predict(ppr, matManager, manager);
            try {
                results.add(decoderModel.predict(encodeResult));
            } finally {
                // Ensure the encoder result (3 tensors + OrtSession.Result) is released even
                // if decoding throws. On the normal path the decoder already closes it, so this
                // is an idempotent no-op there.
                IOUtil.close(encodeResult);
            }
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(new UnirecProcessor().process(matManager, rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        int remaining = refCount.decrementAndGet();
        if (remaining > 0) {
            return;
        }
        if (remaining < 0) {
            refCount.set(0);
            return;
        }
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        INSTANCE_CACHE.remove(key);
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
    }
}
