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
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.OnnxUtil;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class GraniteDoclingEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;

    public GraniteDoclingEncoderModel(final String modelFile,
                                      final int gpuIndex,
                                      final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][][] predict(NDArray not_expanded_pixel_values, NDArray not_expanded_pixel_attention_mask) throws OrtException {
        NDArray pixel_values = not_expanded_pixel_values.expandDims(0);
        NDArray pixel_attention_mask = not_expanded_pixel_attention_mask.expandDims(0);
        NDArray pixel_attention_mask_expand_bool = pixel_attention_mask.toType(DataType.BOOLEAN, false);
        FloatBuffer pixel_values_buffer = pixel_values.toByteBuffer().asFloatBuffer();
        ByteBuffer pixel_attention_mask_buffer = pixel_attention_mask_expand_bool.toByteBuffer();
        long[] pixel_values_shape = pixel_values.getShape().getShape();
        long[] pixel_attention_mask_shape = pixel_attention_mask_expand_bool.getShape().getShape();
        OnnxTensor pixel_values_tensor = OnnxTensor.createTensor(env, pixel_values_buffer, pixel_values_shape);
        OnnxTensor pixel_attention_mask_tensor = OnnxTensor.createTensor(env, pixel_attention_mask_buffer,
                pixel_attention_mask_shape, OnnxJavaType.BOOL);

        Map<String, OnnxTensor> inputs = Map.of(
                "pixel_values", pixel_values_tensor,
                "pixel_attention_mask", pixel_attention_mask_tensor);
        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
        if (optinalResult.isPresent()) {
            Object result = optinalResult.get().getValue();
            onnxResult.close();
            if (result instanceof float[][][] fr) {
                return new float[][][][] {fr};
            } else {
                return (float[][][][]) result;
            }
        }
        onnxResult.close();
        not_expanded_pixel_values.close();
        not_expanded_pixel_attention_mask.close();
        pixel_values.close();
        pixel_attention_mask.close();
        pixel_attention_mask_expand_bool.close();
        OnnxUtil.closeTensors(inputs);
        return null;
    }

}
