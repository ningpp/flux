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
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ByteDanceDolphinElementModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "Dolphin",
            "Dolphin-1.5"
    );

    private final DolphinEncoderModel encoderModel;
    private final DolphinDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final DolphinPreProcessor preProcessor;

    public ByteDanceDolphinElementModel(final String modelRootDir,
                                        final String modelName,
                                        final int gpuIndex,
                                        final OrtEnvironment env,
                                        final boolean skipSpecialTokens) {
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
                    gpuIndex, env, 4096, 1, 2, tokenizer, skipSpecialTokens);
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
        try {
            String task_prompt = "<s>" + prompt + " <Answer/>";
            long[] decoder_input_ids = tokenizer.encode(task_prompt, false, false).getIds();
            try (OnnxTensor encodeResult = encoderModel.predictOnnxTensor(PreProcessResult.getMats(images))) {
                return decoderModel.predict(prompt, encodeResult, decoder_input_ids, manager);
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
