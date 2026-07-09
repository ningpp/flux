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

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.OnnxSessionUtil;

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
    private final int numHiddenLayers = 30;
    private final int numKeyValueHeads = 3;
    private final int headDim = 64;

    public GraniteDoclingDecoderModel(final String modelFile,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final OnnxJavaType dtype) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public GraniteDoclingDecodeResult predict(OnnxTensor inputsEmbeds,
                                              long[] attentionMask,
                                              Map<String, OnnxTensor> pastKeyValues) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>(2 + numHiddenLayers * 2);
        inputs.put("inputs_embeds", inputsEmbeds);

        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attentionMask),
                new long[]{1, attentionMask.length}
        );
        inputs.put("attention_mask", attentionMaskTensor);

        Map<String, OnnxTensor> emptyPastKeyValues = new HashMap<>(numHiddenLayers * 2);
        if (pastKeyValues == null) {
            for (int layer = 0; layer < numHiddenLayers; layer++) {
                for (String kv : new String[]{"key", "value"}) {
                    String key = String.format("past_key_values.%d.%s", layer, kv);
                    OnnxTensor emptyTensor = OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(new float[0]),
                            new long[]{1, numKeyValueHeads, 0, headDim}
                    );
                    emptyPastKeyValues.put(key, emptyTensor);
                    inputs.put(key, emptyTensor);
                }
            }
        } else {
            for (Map.Entry<String, OnnxTensor> entry : pastKeyValues.entrySet()) {
                String key = entry.getKey();
                if (key.startsWith("present.")) {
                    key = "past_key_values." + key.substring("present.".length());
                }
                inputs.put(key, entry.getValue());
            }
        }

        Result onnxResult;
        try {
            onnxResult = session.run(inputs, outputNames);
        } finally {
            attentionMaskTensor.close();
            emptyPastKeyValues.values().forEach(t -> {
                try {
                    if (t != null && !t.isClosed()) {
                        t.close();
                    }
                } catch (Exception ignore) {
                }
            });
        }

        Optional<OnnxValue> optinalResult = onnxResult.get("logits");
        if (optinalResult.isPresent()) {
            float[][][] logits = (float[][][]) optinalResult.get().getValue();
            Map<String, OnnxTensor> present_key_values = new HashMap<>(numHiddenLayers * 2);
            for (String outputName : outputNames) {
                if (!"logits".equals(outputName)) {
                    present_key_values.put(outputName, (OnnxTensor) onnxResult.get(outputName).get());
                }
            }
            return new GraniteDoclingDecodeResult(logits, present_key_values, onnxResult);
        }
        onnxResult.close();
        return null;
    }

}
