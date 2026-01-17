// this code is convert from  https://github.com/breezedeus/Pix2Text
// Pix2Text IS Licensed under the MIT License
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
package io.github.flux.formula.pix2text;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class Pix2TextFormulaRecognitionPredictor extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    public static final List<String> MODEL_NAMES = List.of(
            "pix2text-mfr",
            "pix2text-mfr-1.5"
    );

    private final Pix2TextEncoderModel encoderModel;
    private final Pix2TextDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    public Pix2TextFormulaRecognitionPredictor(final String modelRootDir,
                                               final String modelName,
                                               final int gpuIndex,
                                               final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported pix2text model: " + modelName);
        }

        int maxLength = 513;
        long decoderStartTokenId = 2;
        if (modelName.equals("pix2text-mfr-1.5")) {
            maxLength = 1025;
            decoderStartTokenId = 1;
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        this.encoderModel = new Pix2TextEncoderModel(
                new File(modelDir, "encoder_model.onnx").getAbsolutePath(),
                gpuIndex, env, new Pix2TextPreProcessor()
        );

        try {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new Pix2TextDecoderModel(
                    new File(modelDir, "decoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, maxLength, 0, 2, decoderStartTokenId, tokenizer);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new PreProcessResult(null, new Pix2TextPreProcessor().process(matManager, rgbMat, manager));
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> inputNDArrays, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        try {
            float[][][] encoderHiddenStates = encoderModel.batchPredict(PreProcessResult.getNDArrays(inputNDArrays));
            return decoderModel.batchPredict(encoderHiddenStates);
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
