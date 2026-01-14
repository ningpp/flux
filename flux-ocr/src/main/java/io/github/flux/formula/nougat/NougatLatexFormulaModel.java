// this code is convert from https://github.com/NormXU/nougat-latex-ocr
// nougat-latex-ocr's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.nougat;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
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
import java.util.Set;

public class NougatLatexFormulaModel extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "nougat-latex-base"
    );

    private final NougatLatexEncoderModel encoderModel;
    private final NougatLatexDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final NougatImageProcessor preProcessor;

    public NougatLatexFormulaModel(final String modelRootDir,
                                   final String modelName,
                                   final int gpuIndex,
                                   final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported nougat latex model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.preProcessor = new NougatImageProcessor();
            this.encoderModel = new NougatLatexEncoderModel(new File(modelDir, "encoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, preProcessor);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new NougatLatexDecoderModel(new File(modelDir, "decoder_model.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096, 1, 2, 0, tokenizer);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        NDArray ndArray = preProcessor.process(matManager, rgbMat, manager);
        return new PreProcessResult(null, ndArray);
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> batch, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        try {
            float[][][] encoderResults = encoderModel.batchPredict(PreProcessResult.getNDArrays(batch), manager);
            return decoderModel.batchPredict(encoderResults, manager);
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
