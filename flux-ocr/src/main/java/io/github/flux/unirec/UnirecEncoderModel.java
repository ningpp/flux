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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;

import java.util.Map;

public class UnirecEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public UnirecEncoderModel(final String modelFile,
                              final int gpuIndex,
                              final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public UnirecEncoderModelPredictResult predict(PreProcessResult ppr, MatManager matManager, NDManager ndManager) {
        try {
            NDList ndList = new NDList();
            ndList.add(ImageUtil.toChannalNDArrayFloat(matManager, ppr.mat(), ndManager));
            NDArray inputNdArray = NDArrays.stack(ndList);
            IOUtil.close(ppr.mat());
            long[] shape = inputNdArray.getShape().getShape();
            var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
            Result result = session.run(Map.of("pixel_values", onnxInput));
            IOUtil.close(onnxInput);
            IOUtil.close(inputNdArray);
            IOUtil.close(ndList);
            return new UnirecEncoderModelPredictResult(
                    (OnnxTensor) result.get(0),
                    (OnnxTensor) result.get(1),
                    (OnnxTensor) result.get(2));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

}
