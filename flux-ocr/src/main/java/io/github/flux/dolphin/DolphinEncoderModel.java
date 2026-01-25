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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class DolphinEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public DolphinEncoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            this.inputName = List.copyOf(session.getInputNames()).getFirst();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public OnnxTensor predictOnnxTensor(List<Mat> inputMats) throws OrtException {
        try (OnnxTensor onnxInput = ImageUtil.matToOnnxTensor(inputMats, env)) {
            OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput));
            return (OnnxTensor) onnxResult.get(0);
        }
    }

}
