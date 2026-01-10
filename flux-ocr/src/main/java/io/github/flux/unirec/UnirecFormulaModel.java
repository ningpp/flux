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
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class UnirecFormulaModel extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {

    private final UnirecPredictor predictor;

    public UnirecFormulaModel(UnirecPredictor predictor) {
        this.predictor = predictor;
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> mats, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, manager, extraParameters).stream()
                .map(r -> new FormulaRecognitionResult(List.of(r.text()), r.tokens(), r.score())).toList();
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return predictor.processRgb(rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

}
