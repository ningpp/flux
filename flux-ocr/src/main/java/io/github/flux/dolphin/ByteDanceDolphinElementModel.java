// this code is convert from  https://github.com/bytedance/Dolphin/blob/v1.5
// Dolphin v1.5 IS Licensed under the MIT License
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
package io.github.flux.dolphin;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.InstanceKey;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.TableModel;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ByteDanceDolphinElementModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "Dolphin",
            "Dolphin-1.5"
    );

    // Shared instance cache to avoid creating multiple expensive model instances
    private static final Map<InstanceKey, ByteDanceDolphinElementModel> INSTANCE_CACHE = new ConcurrentHashMap<>();

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, ByteDanceDolphinFormulaModel::new);
        TableModel.getRegistry().register(MODEL_NAMES, ByteDanceDolphinTableModel::new);
    }

    private final DolphinEncoderModel encoderModel;
    private final DolphinDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final DolphinPreProcessor preProcessor;

    /**
     * Gets a shared instance of ByteDanceDolphinElementModel for the given configuration.
     * If an instance with the same configuration already exists, it will be reused.
     * This is important because the model is expensive to create (loads ONNX models).
     * The skipSpecialTokens parameter is now passed at prediction time, not construction time,
     * allowing the same model instance to be shared between formula and table tasks.
     *
     * @param modelRootDir the root directory containing model files
     * @param modelName the name of the model
     * @param gpuIndex the GPU index to use (-1 for CPU)
     * @param env the ONNX runtime environment
     * @return a shared instance of the model
     */
    public static ByteDanceDolphinElementModel getSharedInstance(final String modelRootDir,
                                                                  final String modelName,
                                                                  final int gpuIndex,
                                                                  final OrtEnvironment env) {
        InstanceKey key = new InstanceKey(modelRootDir, modelName, gpuIndex);
        return INSTANCE_CACHE.computeIfAbsent(key, k ->
            new ByteDanceDolphinElementModel(modelRootDir, modelName, gpuIndex, env));
    }

    /**
     * Private constructor - use getSharedInstance() to obtain instances.
     */
    private ByteDanceDolphinElementModel(final String modelRootDir,
                                         final String modelName,
                                         final int gpuIndex,
                                         final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported pix2text model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.preProcessor = new DolphinPreProcessor();
            this.encoderModel = new DolphinEncoderModel(new File(modelDir, "encoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new DolphinDecoderModel(new File(modelDir, "decoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096, 1, 2, tokenizer);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(preProcessor.process(matManager, rgbMat), null);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> images, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        String prompt = String.valueOf(extraParameters.getOrDefault("prompt", "Read text in the image."));
        boolean skipSpecialTokens = (boolean) extraParameters.getOrDefault("skipSpecialTokens", false);
        try {
            String task_prompt = "<s>" + prompt + " <Answer/>";
            long[] decoder_input_ids = tokenizer.encode(task_prompt, false, false).getIds();
            try (OnnxTensor encodeResult = encoderModel.predictOnnxTensor(PreProcessResult.getMats(images))) {
                return decoderModel.predict(prompt, encodeResult, decoder_input_ids, manager, skipSpecialTokens);
            }
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
