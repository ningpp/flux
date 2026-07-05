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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxUtil;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Pix2TextEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Pix2TextPreProcessor preProcessor;

    public Pix2TextEncoderModel(final String modelFile,
                                final int gpuIndex,
                                final OrtEnvironment env,
                                Pix2TextPreProcessor preProcessor) {
        try {
            this.env = env;
            this.preProcessor = preProcessor;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] batchPredict(List<NDArray> inputNDArrays) throws OrtException {
        NDList ndList = new NDList();
        NDArray inputNdArray = null;
        OnnxTensor onnxInput = null;
        OrtSession.Result onnxResult = null;
        try {
            ndList.addAll(inputNDArrays);
            inputNdArray = NDArrays.stack(ndList);
            inputNDArrays.forEach(IOUtil::close);
            long[] shape = inputNdArray.getShape().getShape();
            var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
            IOUtil.close(inputNdArray);
            inputNdArray = null;
            IOUtil.close(ndList);
            ndList = null;
            Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
            for (String inputName : inputNames) {
                inputs.put(inputName, onnxInput);
            }
            onnxInput = null; // ownership transferred to inputs map
            onnxResult = session.run(inputs);
            OnnxUtil.closeTensors(inputs);
            float[][][] encodeResultFloats = (float[][][]) onnxResult.get(0).getValue();
            IOUtil.close(onnxResult);
            onnxResult = null;
            return encodeResultFloats;
        } finally {
            IOUtil.close(onnxResult);
            IOUtil.close(onnxInput);
            IOUtil.close(inputNdArray);
            IOUtil.close(ndList);
        }
    }

}
