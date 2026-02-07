// this code is convert from https://github.com/OleehyO/TexTeller
// TexTeller's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.texteller;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.TextResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import io.github.flux.model.FormulaRecognitionModel;
import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class TexTellerPredictor extends BatchPredictor<PreProcessResult, TextResult> {

    public static final List<String> MODEL_NAMES = List.of(
            "TexTeller"
    );

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, TexTellerPredictor::new);
    }

    private final TexTellerEncoderModel encoderModel;
    private final TexTellerDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    private final TexTellerProcessor preProcessor = new TexTellerProcessor();

    public TexTellerPredictor(final String modelRootDir,
                              final String modelName,
                              final int gpuIndex,
                              final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported TexTeller model: " + modelName);
        }
        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new TexTellerEncoderModel(
                    new File(modelDir, "encoder_model.onnx").getAbsolutePath(), gpuIndex, env
            );
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new TexTellerDecoderModel(
                    new File(modelDir, "decoder_model_merged.onnx").getAbsolutePath(),
                    gpuIndex, env, 1024, 1, 2, 0, tokenizer);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> mats,
                                                         MatManager matManager, NDManager ndManager,
                                                         Map<String, Object> extraParameters) {
        try {
            float[][][] last_hidden_states = encoderModel.batchPredict(PreProcessResult.getMats(mats), matManager, ndManager);
            return decoderModel.batchPredict(last_hidden_states, ndManager);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(preProcessor.process(matManager, rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(tokenizer);
    }

}
