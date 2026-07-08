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
import io.github.flux.util.OnnxSessionUtil;
import ai.onnxruntime.OrtSession.Result;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
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
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);

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

    public TextResult predict(UnirecEncoderModelPredictResult encodeResult) {
        try {
            int num_decoder_layers = 6;
            int num_heads = 6;
            int head_dim = 128;
            int maxTokens = 2048;
            long[] generated_ids = new long[maxTokens + 1];
            generated_ids[0] = bos_token_id;

            // Initialize empty past_key_values for first step
            // Shape: [batch_size, num_heads, 0, head_dim]
            List<Pair<OnnxTensor, OnnxTensor>> initialPastKeyValues = new ArrayList<>();
            int batch_size = (int) encodeResult.hiddenStates().getInfo().getShape()[0];
            for (int i = 0; i < num_decoder_layers; i++) {
                float[][][][] empty_key = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
                float[][][][] empty_value = ArrayUtil.createZeros(batch_size, num_heads, 0, head_dim);
                initialPastKeyValues.add(Pair.of(ArrayUtil.createOnnxTensor(empty_key, env),
                        ArrayUtil.createOnnxTensor(empty_value, env)));
            }

            OnnxTensor crossKOnnxTensor = encodeResult.crossK();
            OnnxTensor crossVOnnxTensor = encodeResult.crossV();
            int generatedTokens = 0;
            // Holds the Result of the most recent decode step so its native handle can be
            // released once the present_key_values it produced are no longer referenced.
            DecodeStepResult dsr = null;
            List<Pair<OnnxTensor, OnnxTensor>> past_key_values = initialPastKeyValues;
            for (int i = 0; i < maxTokens; i++) {
                long current_token = generated_ids[i];
                // Decode step closes only the newly created input tensors (input_ids,
                // position_ids). The input past_key_values are owned by the previous step's
                // Result and are closed when that Result is released below.
                DecodeStepResult next = decode_step(
                        current_token,
                        i,
                        crossKOnnxTensor,
                        crossVOnnxTensor,
                        past_key_values,
                        pad_token_id,
                        num_decoder_layers);
                // The previous step's Result owns its output tensors, which served as the
                // current step's past_key_values input. Now that the current step has run,
                // those tensors are no longer needed and are released with the Result.
                if (dsr != null) {
                    IOUtil.close(dsr.result());
                }
                dsr = next;
                past_key_values = dsr.past_key_values();
                float[][][] logits = dsr.logits();
                // Get next token
                long next_token_id = getNextTokenId(logits);
                generated_ids[i+1] = next_token_id;

                // Check for EOS
                if (next_token_id == eos_token_id) {
                    generatedTokens = i+1;
                    break;
                }
            }
            if (generatedTokens == 0) {
                generatedTokens = maxTokens + 1;
            }
            long[] ids = new long[generatedTokens];
            System.arraycopy(generated_ids, 0, ids, 0, generatedTokens);

            // Clean up: the final Result owns its output tensors (final past_key_values),
            // so closing it releases them. The initial empty past_key_values are owned by us.
            if (dsr != null) {
                IOUtil.close(dsr.result());
            }
            for (var pastKvPair : initialPastKeyValues) {
                IOUtil.close(pastKvPair.getLeft());
                IOUtil.close(pastKvPair.getRight());
            }
            // encodeResult is owned by the caller (UnirecPredictor) and closed in its finally block.
            return new TextResult(clean_special_tokens(decodeTokenIds(ids)), ids, -1f);
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
            if (intId >= id2tokens.length) {
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
                                         OnnxTensor crossKOnnxTensor,
                                         OnnxTensor crossVOnnxTensor,
                                         List<Pair<OnnxTensor, OnnxTensor>> pastKeyValues,
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
        decoder_inputs.put("cross_k", crossKOnnxTensor);
        decoder_inputs.put("cross_v", crossVOnnxTensor);
        for (int i = 0; i < pastKeyValues.size(); i++) {
            OnnxTensor keyTensor = pastKeyValues.get(i).getLeft();
            OnnxTensor valueTensor = pastKeyValues.get(i).getRight();
            decoder_inputs.put("past_key_" + i, keyTensor);
            decoder_inputs.put("past_value_" + i, valueTensor);
        }

        Result result = session.run(decoder_inputs);
        // Close only the newly created input tensors (input_ids, position_ids).
        // Do NOT close cross_k/cross_v - they are owned by the caller (the encoder result)
        // and reused every step. Do NOT close past_key_values - they are outputs of the
        // previous step's Result and are released when that Result is closed by the caller.
        IOUtil.close(inputIdsTensor);
        IOUtil.close(positionIdsTensor);
        float[][][] logits = (float[][][]) result.get(0).getValue();
        List<Pair<OnnxTensor, OnnxTensor>> present_key_values = new ArrayList<>();
        // Extract present_key_values
        for (int i = 0; i < num_decoder_layers; i++) {
            present_key_values.add(Pair.of(
                    (OnnxTensor) result.get(1 + i * 2),
                    (OnnxTensor) result.get(1 + i * 2 + 1)));
        }
        // The Result is returned to the caller and closed later (once its present_key_values
        // outputs are no longer referenced), so its native handle is not leaked.
        return new DecodeStepResult(logits, present_key_values, result);
    }

    private record DecodeStepResult(float[][][] logits,
                                    List<Pair<OnnxTensor, OnnxTensor>> past_key_values,
                                    OrtSession.Result result) {

    }

}
