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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;

import java.util.List;
import java.util.Map;

public class NougatLatexEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public NougatLatexEncoderModel(final String modelFile,
                                   final int gpuIndex,
                                   final OrtEnvironment env,
                                   final NougatImageProcessor preProcessor) {
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

    public OnnxTensor batchPredict(List<NDArray> inputNDArrays, NDManager manager) throws OrtException {
        NDList ndList = new NDList();
        ndList.addAll(inputNDArrays);
        NDArray inputNdArray = NDArrays.stack(ndList);
        inputNDArrays.forEach(IOUtil::close);
        long[] shape = inputNdArray.getShape().getShape();
        var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        IOUtil.close(inputNdArray);
        IOUtil.close(ndList);
        try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape)) {
            OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput));
            return (OnnxTensor) onnxResult.get(0);
        }
    }

}
