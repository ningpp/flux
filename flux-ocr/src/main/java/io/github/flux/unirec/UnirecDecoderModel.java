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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.github.flux.dolphin.DolphinElementResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import org.apache.commons.lang3.tuple.Pair;

import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class UnirecDecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final long bos_token_id = 0;
    private final long eos_token_id = 2;
    private final long pad_token_id = 1;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String[] id2tokens;

    public UnirecDecoderModel(final String modelFile,
                              final String tokenFile,
                              final int gpuIndex,
                              final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            JsonElement jsonElement = JsonParser.parseString(Files.readString(Paths.get(tokenFile), StandardCharsets.UTF_8));
            JsonObject id_to_token_json_obj = jsonElement.getAsJsonObject().getAsJsonObject("id_to_token");
            Set<Entry<String, JsonElement>> id_to_token_entries = id_to_token_json_obj.entrySet();
            List<Pair<Integer, String>> pairs = new ArrayList<>();
            for (var entry : id_to_token_entries) {
                pairs.add(Pair.of(Integer.valueOf(entry.getKey()), entry.getValue().getAsString()));
            }
            pairs.sort(Comparator.comparing(Pair::getKey));

            id2tokens = new String[pairs.size()];
            for (int i = 0; i < id2tokens.length; i++) {
                id2tokens[i] = pairs.get(i).getRight();
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public DolphinElementResult predict(UnirecEncoderModelPredictResult encodeResult) {
        try {
            int num_decoder_layers = 6;
            int num_heads = 6;
            int head_dim = 128;
            int maxTokens = 2048;
            long[] generated_ids = new long[maxTokens];
            generated_ids[0] = bos_token_id;

            // Initialize empty past_key_values for first step
            // Shape: [batch_size, num_heads, 0, head_dim]
            List<Pair<float[][][][], float[][][][]>> past_key_values = new ArrayList<>();
            int batch_size = encodeResult.hiddenStates().length;
            for (int i = 0; i < num_decoder_layers; i++) {
                float[][][][] empty_key = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
                float[][][][] empty_value = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
                past_key_values.add(Pair.of(empty_key, empty_value));
            }

            int generatedTokens = 0;
            for (int i = 0; i < maxTokens; i++) {
                long current_token = generated_ids[i];
                // Decode step
                DecodeStepResult dsr = decode_step(
                        current_token,
                        i,
                        encodeResult.crossK(),
                        encodeResult.crossV(),
                        past_key_values,
                        pad_token_id,
                        num_decoder_layers);
                float[][][] logits = dsr.logits;
                past_key_values = dsr.past_key_values;
                // Get next token
                long next_token_id = getNextTokenId(logits);
                generated_ids[i+1] = next_token_id;

                // Check for EOS
                if (next_token_id == eos_token_id) {
                    generatedTokens = i+1;
                    break;
                }
            }
            long[] ids = new long[generatedTokens];
            System.arraycopy(generated_ids, 0, ids, 0, generatedTokens);
            return new DolphinElementResult(clean_special_tokens(decodeTokenIds(ids)), ids, -1f);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private String decodeTokenIds(long[] ids) {
        StringBuilder builder = new StringBuilder();
        for (long id : ids) {
            if (id == bos_token_id || id == eos_token_id || id == pad_token_id) {
                continue;
            }
            int intId = Math.toIntExact(id);
            if (intId > id2tokens.length) {
                continue;
            }
            builder.append(id2tokens[intId]);
        }
        return builder.toString();
    }

    // Clean special tokens from decoded text.
    private String clean_special_tokens(String text) {
        String str = text.replace("Ġ", " ").replace("Ċ", "\n")
                .replaceAll("-\\<\\|sn\\|\\>", "")
                .replaceAll(" \\<\\|sn\\|\\>", " ")
                .replaceAll("\\<\\|sn\\|\\>", " ")
                .replaceAll("\\<\\|unk\\|\\>", "")
                .replaceAll("\\<s\\>", "")
                .replaceAll("\\<\\/s\\>", "")
                .replaceAll("\uffff", "")
                .replaceAll("_{4,}", "___")
                .replaceAll("\\.{4,}", "...");
        if (str.startsWith("\\[") && str.endsWith("\\]")) {
            // remove for formula
            return str.substring(2, str.length()-2);
        } else {
            return str;
        }
    }

    private int getNextTokenId(float[][][] logits) {
        float[] lastLogits = logits[0][logits[0].length - 1];
        int maxIndex = 0;
        float max = lastLogits[0];

        // 手动展开循环或使用向量化操作
        for (int i = 1; i < lastLogits.length; i++) {
            float val = lastLogits[i];
            if (val > max) {
                max = val;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    // Unified decoder step with or without cache.
    private DecodeStepResult decode_step(long input_id,
                                         int pastLength,
                                         float[][][][][] cross_k,
                                         float[][][][][] cross_v,
                                         List<Pair<float[][][][], float[][][][]>> pastKeyValues,
                                         long paddingIdx,
                                         int num_decoder_layers) throws OrtException {

        // Prepare inputs
        LongBuffer inputIds = LongBuffer.wrap(new long[] { input_id });
        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIds, new long[] {1, 1});

        // Use M2M100's position ID calculation with past_key_values_length
        LongBuffer positionIds = LongBuffer.wrap(new long[] { paddingIdx + 1 + pastLength });
        OnnxTensor positionIdsTensor = OnnxTensor.createTensor(env, positionIds, new long[] {1, 1});

        Map<String, OnnxTensor> decoder_inputs = new HashMap<>();
        decoder_inputs.put("input_ids", inputIdsTensor);
        decoder_inputs.put("position_ids", positionIdsTensor);
        decoder_inputs.put("cross_k", ArrayUtil.createOnnxTensor(cross_k, env));
        decoder_inputs.put("cross_v", ArrayUtil.createOnnxTensor(cross_v, env));
        for (int i = 0; i < pastKeyValues.size(); i++) {
            decoder_inputs.put("past_key_" + i, ArrayUtil.createOnnxTensor(pastKeyValues.get(i).getLeft(), env));
            decoder_inputs.put("past_value_" + i, ArrayUtil.createOnnxTensor(pastKeyValues.get(i).getRight(), env));
        }

        Result result = session.run(decoder_inputs);
        float[][][] logits = (float[][][]) result.get(0).getValue();
        List<Pair<float[][][][], float[][][][]>> present_key_values = new ArrayList<>();
        // Extract present_key_values
        for (int i = 0; i < num_decoder_layers; i++) {
            float[][][][] key = (float[][][][]) result.get(1 + i * 2).getValue();
            float[][][][] value = (float[][][][]) result.get(1 + i * 2 + 1).getValue();
            present_key_values.add(Pair.of(key, value));
        }
        return new DecodeStepResult(logits, present_key_values);
    }

    private record DecodeStepResult(float[][][] logits, List<Pair<float[][][][], float[][][][]>> past_key_values) {

    }

}
