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
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class GraniteDoclingDecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> outputNames;
    private final OnnxJavaType dtype;

    public GraniteDoclingDecoderModel(final String modelFile,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final OnnxJavaType dtype) {
        try {
            this.env = env;
            this.dtype = dtype;
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

    public GraniteDoclingDecodeResult predict(NDArray inputs_embeds, NDArray attention_mask,
                                              Map<String, NDArray> past_key_values) throws OrtException {
        // don't close attention_mask
        FloatBuffer inputs_embeds_buffer = inputs_embeds.toByteBuffer().asFloatBuffer();
        LongBuffer attention_mask_buffer = attention_mask.toByteBuffer().asLongBuffer();
        long[] inputs_embeds_shape = inputs_embeds.getShape().getShape();
        long[] attention_mask_shape = attention_mask.getShape().getShape();
        OnnxTensor inputs_embeds_tensor = OnnxTensor.createTensor(env, inputs_embeds_buffer, inputs_embeds_shape);
        OnnxTensor attention_mask_tensor = OnnxTensor.createTensor(env, attention_mask_buffer, attention_mask_shape);

        Map<String, OnnxTensor> inputs = new HashMap<>(2);
        inputs.put("inputs_embeds", inputs_embeds_tensor);
        inputs.put("attention_mask", attention_mask_tensor);
        Set<Map.Entry<String, NDArray>> past_key_values_entrySet = past_key_values.entrySet();
        for (var entry : past_key_values_entrySet) {
            NDArray past_key_value = entry.getValue();
            long[] past_key_value_shape = past_key_value.getShape().getShape();
            OnnxTensor past_key_value_tensor;
            if (OnnxJavaType.FLOAT16 == dtype) {
                NDArray past_key_value_short = entry.getValue().toType(DataType.FLOAT16, true);
                var past_key_value_buffer = past_key_value_short.toByteBuffer().asShortBuffer();
                past_key_value_tensor = OnnxTensor.createTensor(env, past_key_value_buffer, past_key_value_shape,
                        OnnxJavaType.FLOAT16);
                past_key_value_short.close();
            } else {
                var past_key_value_buffer = past_key_value.toByteBuffer().asFloatBuffer();
                past_key_value_tensor = OnnxTensor.createTensor(env, past_key_value_buffer, past_key_value_shape);
            }
            inputs.put(entry.getKey(), past_key_value_tensor);
        }
        past_key_values.forEach((_, pkv) -> IOUtil.close(pkv));

        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get("logits");
        if (optinalResult.isPresent()) {
            float[][][] logits = (float[][][]) optinalResult.get().getValue();
            Map<String, float[][][][]> present_key_values = new HashMap<>();
            for (String outputName : outputNames) {
                if (!"logits".equals(outputName)) {
                    present_key_values.put(outputName, (float[][][][]) onnxResult.get(outputName).get().getValue());
                }
            }
            onnxResult.close();
            inputs_embeds.close();
            OnnxUtil.closeTensors(inputs);
            return new GraniteDoclingDecodeResult(logits, present_key_values);
        }
        onnxResult.close();
        inputs_embeds.close();
        OnnxUtil.closeTensors(inputs);
        return null;
    }

}
