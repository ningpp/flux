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
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxUtil;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class Pix2TextEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
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
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] batchPredict(List<NDArray> inputNDArrays, NDManager manager) {
        try {
            return _batchPredict(inputNDArrays, manager);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] _batchPredict(List<NDArray> inputNDArrays, NDManager manager) throws OrtException {
        NDList ndList = new NDList();
        ndList.addAll(inputNDArrays);
        NDArray inputNdArray = NDArrays.stack(ndList);
        inputNDArrays.forEach(IOUtil::close);
        long[] shape = inputNdArray.getShape().getShape();
        var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        IOUtil.close(inputNdArray);
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
        IOUtil.close(onnxResult);
        OnnxUtil.closeTensors(inputs);
        return encodeResultFloats;
    }

    public  float[][][] predict(MatManager matManager, Mat srcImage, NDManager manager) {
        try {
            NDArray inputNdArray = preProcessor.process(matManager, srcImage, manager);
            NDArray expandResult = inputNdArray.expandDims(0);
            FloatBuffer dataBuffer = expandResult.toByteBuffer().asFloatBuffer();
            long[] shape = expandResult.getShape().getShape();
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
            for (String inputName : inputNames) {
                inputs.put(inputName, onnxInput);
            }
            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            if (optinalResult.isPresent()) {
                float[][][] encodeResultFloats = (float[][][]) optinalResult.get().getValue();
                return encodeResultFloats;
            }
            onnxResult.close();
            expandResult.close();
            inputNdArray.close();
            OnnxUtil.closeTensors(inputs);
            return null;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

}
