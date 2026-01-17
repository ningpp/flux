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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class TexTellerEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public TexTellerEncoderModel(final String modelFile,
                                 final int gpuIndex,
                                 final OrtEnvironment env) throws OrtException {
        this.env = env;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        if (gpuIndex > -1) {
            options.addCUDA(gpuIndex);
        }
        this.session = env.createSession(modelFile, options);
    }

    public float[][][] batchPredict(List<Mat> inputMats, MatManager matManager, NDManager manager) throws OrtException {
        NDList ndList = new NDList();
        for (Mat mat : inputMats) {
            ndList.add(ImageUtil.toChannalNDArrayFloat(matManager, mat, manager));
        }
        NDArray inputNdArray = NDArrays.stack(ndList);
        long[] shape = inputNdArray.getShape().getShape();
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        IOUtil.close(inputNdArray);
        IOUtil.close(ndList);
        OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        Map<String, OnnxTensor> inputs = Map.of("pixel_values", onnxInput);
        OrtSession.Result onnxResult = session.run(inputs);
        OnnxValue onnxValue = onnxResult.get(0);
        float[][][] results = (float[][][]) onnxValue.getValue();
        IOUtil.close(onnxValue);
        IOUtil.close(onnxInput);
        return results;
    }

}
