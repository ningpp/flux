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

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class ByteDanceDolphinFormulaModel extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    private final ByteDanceDolphinElementModel model;

    public ByteDanceDolphinFormulaModel(ByteDanceDolphinElementModel model) {
        this.model = model;
    }

    public ByteDanceDolphinFormulaModel(final String modelRootDir,
                                        final String modelName,
                                        final int gpuIndex,
                                        final OrtEnvironment env) {
        this.model = new ByteDanceDolphinElementModel(modelRootDir, modelName, gpuIndex, env, true);
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        extraParameters.put("prompt", "Read formula in the image.");
        var elementResults = model.doBatchPredict(mats, matManager, manager, extraParameters);
        return elementResults.stream().map(r -> new FormulaRecognitionResult(
                List.of(removeDollar(r.text())), r.tokens(), r.score())).toList();
    }

    private String removeDollar(String text) {
        if (text.startsWith("$$") && text.endsWith("$$")) {
            return text.substring(2, text.length()-2);
        } else {
            return text;
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return model.processRgb(matManager, rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(model);
    }
}
