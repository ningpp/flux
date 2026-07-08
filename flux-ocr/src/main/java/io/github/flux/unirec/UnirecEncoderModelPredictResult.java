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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;

public class UnirecEncoderModelPredictResult implements AutoCloseable {
    private final MatManager matManager;
    private final OrtSession.Result result;
    private final OnnxTensor hiddenStates;
    private final OnnxTensor crossK;
    private final OnnxTensor crossV;

    public UnirecEncoderModelPredictResult(MatManager matManager,
                                           OrtSession.Result result,
                                           OnnxTensor hiddenStates,
                                           OnnxTensor crossK,
                                           OnnxTensor crossV) {
        this.matManager = matManager;
        this.result = result;
        this.hiddenStates = hiddenStates;
        this.crossK = crossK;
        this.crossV = crossV;
    }

    public OnnxTensor hiddenStates() {
        return hiddenStates;
    }

    public OnnxTensor crossK() {
        return crossK;
    }

    public OnnxTensor crossV() {
        return crossV;
    }

    @Override
    public void close() {
        // Closing the Result also closes the OnnxTensors extracted from it.
        // Do not close hiddenStates/crossK/crossV separately to avoid
        // "Closing an already closed tensor" warnings.
        matManager.release(result);
    }
}
