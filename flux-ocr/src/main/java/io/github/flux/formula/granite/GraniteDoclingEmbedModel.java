// this code is convert from https://github.com/huggingface/transformers
// transformers's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.granite;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.OnnxUtil;

import java.nio.LongBuffer;
import java.util.Map;
import java.util.Optional;

public class GraniteDoclingEmbedModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public GraniteDoclingEmbedModel(final String modelFile,
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

    public float[][][] predict(NDArray input_ids) throws OrtException {
        // don't close input_ids
        DataType dataType = input_ids.getDataType();
        LongBuffer buffer;
        if (dataType == DataType.INT64) {
            buffer = input_ids.toByteBuffer().asLongBuffer();
        } else {
            NDArray input_ids_long = input_ids.toType(DataType.INT64, false);
            buffer = input_ids_long.toByteBuffer().asLongBuffer();
            input_ids_long.close();
        }
        long[] shape = input_ids.getShape().getShape();
        OnnxTensor tensor = OnnxTensor.createTensor(env, buffer, shape);
        Map<String, OnnxTensor> inputs = Map.of("input_ids", tensor);
        OrtSession.Result onnxResult = session.run(inputs);
        Optional<OnnxValue> optinalResult = onnxResult.get("inputs_embeds");
        if (optinalResult.isPresent()) {
            float[][][] encodeResultFloats = (float[][][]) optinalResult.get().getValue();
            onnxResult.close();
            OnnxUtil.closeTensors(inputs);
            return encodeResultFloats;
        }
        onnxResult.close();
        OnnxUtil.closeTensors(inputs);
        return null;
    }

}
