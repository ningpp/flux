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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.OnnxUtil;
import org.opencv.core.Mat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class DolphinEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
    private final OnnxJavaType dtype;

    public DolphinEncoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               final OnnxJavaType dtype) {
        this.dtype = dtype;
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] batchPredict(List<Mat> inputMats, MatManager matManager, NDManager manager) throws OrtException {
        NDList ndList = new NDList();
        for (Mat mat : inputMats) {
            ndList.add(ImageUtil.toChannalNDArrayFloat(matManager, mat, manager));
        }
        NDArray inputNdArray = NDArrays.stack(ndList);
        inputMats.forEach(Mat::release);
        long[] shape = inputNdArray.getShape().getShape();
        OnnxTensor onnxInput;
        if (dtype == OnnxJavaType.FLOAT) {
            var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        } else {
            NDArray expandResultFloat16 = inputNdArray.toType(DataType.FLOAT16, true);
            var buffer = expandResultFloat16.toByteBuffer().asShortBuffer();
            onnxInput = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16);
            IOUtil.close(expandResultFloat16);
        }
        inputNdArray.close();
        IOUtil.close(ndList);

        Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
        for (String inputName : inputNames) {
            inputs.put(inputName, onnxInput);
        }
        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
        float[][][] encodeResultFloats = null;
        if (optinalResult.isPresent()) {
            encodeResultFloats = (float[][][]) optinalResult.get().getValue();
        }
        onnxResult.close();
        OnnxUtil.closeTensors(inputs);
        return encodeResultFloats;
    }

    public float[][][] predict(MatManager matManager, Mat inputMat, NDManager manager) throws OrtException {
        NDArray inputNdArray = ImageUtil.toChannalNDArrayFloat(matManager, inputMat, manager);
        inputMat.release();
        NDArray expandResult = inputNdArray.expandDims(0);
        inputNdArray.close();
        long[] shape = expandResult.getShape().getShape();
        OnnxTensor onnxInput;
        if (dtype == OnnxJavaType.FLOAT) {
            var dataBuffer = expandResult.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        } else {
            NDArray expandResultFloat16 = expandResult.toType(DataType.FLOAT16, true);
            var buffer = expandResultFloat16.toByteBuffer().asShortBuffer();
            onnxInput = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16);
            IOUtil.close(expandResultFloat16);
        }
        IOUtil.close(expandResult);

        Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
        for (String inputName : inputNames) {
            inputs.put(inputName, onnxInput);
        }
        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
        float[][][] encodeResultFloats = null;
        if (optinalResult.isPresent()) {
            encodeResultFloats = (float[][][]) optinalResult.get().getValue();
        }
        onnxResult.close();
        OnnxUtil.closeTensors(inputs);
        return encodeResultFloats;
    }

}
