// this code is convert from https://github.com/PaddlePaddle/PaddleX
// PaddleX's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.paddle.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.RecognitionResult;
import io.github.flux.exception.FluxException;
import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CTCLabelDecode {

    private final List<String> character;
    private final Map<String, Integer> dict;
    private final boolean reverse;

    public CTCLabelDecode(String yamlFile) {
        this(yamlFile, false, true);
    }

    public CTCLabelDecode(String yamlFile, boolean reverse, boolean useSpaceChar) {
        this.reverse = reverse;
        Yaml yaml = new Yaml();
        try {
            try (InputStream inputStream = Files.newInputStream(Paths.get(yamlFile))) {
                Map<String, Object> config = yaml.load(inputStream);
                Map<String, Object> object = (Map<String, Object>) config.get("PostProcess");
                List<String> characters = (List<String>) object.get("character_dict");
                characters.add(0, "blank");
                if (useSpaceChar) {
                    characters.add(" ");
                }
                character = List.copyOf(characters);
                Map<String, Integer> dicts = new HashMap<>();
                int i = 0;
                for (String c : characters) {
                    dicts.put(c, i++);
                }
                dict = Map.copyOf(dicts);
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private static final List<Integer> IGNORE_TOKENS = List.of(0);

    private List<Integer> get_ignored_tokens() {
        return IGNORE_TOKENS;
    }

    // convert text-index into text-label.
    public List<RecognitionResult> decode(NDArray text_index, NDArray text_prob, boolean is_remove_duplicate) {
        var manager = text_index.getManager();
        List<RecognitionResult> result_list = new ArrayList<>();
        var ignored_tokens = get_ignored_tokens();
        int batch_size = (int) text_index.getShape().getShape()[0];
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            NDArray selection = manager.ones(new Shape(text_index.getShape().getShape()[1]), DataType.BOOLEAN);
            if (is_remove_duplicate) {
                NDIndex slice = new NDIndex("1:");
                // Set all elements in this slice to false
                NDArray idx_array = text_index.get(batch_idx);
                NDArray v1 = idx_array.get(new NDIndex("1:"));
                NDArray v2 = idx_array.get(new NDIndex(":-1"));
                NDArray equalNdArray = v1.eq(v2);
                NDArray comparisonResult = equalNdArray.logicalNot();
                selection.set(slice, comparisonResult);
                equalNdArray.close();
                comparisonResult.close();
                v2.close();
                v1.close();
                idx_array.close();
            }
            for (Integer ignored_token : ignored_tokens) {
                NDArray v = text_index.get(batch_idx);
                NDArray condition = v.neq(ignored_token);
                NDArray andValue = selection.logicalAnd(condition);
                andValue.copyTo(selection);
                andValue.close();
                condition.close();
                v.close();
            }
            List<String> chat_list = new ArrayList<>();
            NDArray index_text_values = text_index.get(batch_idx);
            NDArray text_values = index_text_values.get(selection);
            int size = (int) text_values.size();
            for (int i = 0; i < size; i++) {
                int index = (int) text_values.getLong(i);
                // System.out.println(index);
                // TODO
                if (index < character.size()) {
                    chat_list.add(character.get(index));
                }
            }
            text_values.close();
            index_text_values.close();

            NDArray conf_list;
            if (text_prob != null) {
                NDArray idx_prob = text_prob.get(batch_idx);
                conf_list = idx_prob.get(selection);
                idx_prob.close();
            } else {
                conf_list = manager.ones(new Shape(selection.size()));
            }
            if (conf_list.isEmpty()) {
                conf_list = manager.create(0);
            }
            String text = String.join("", chat_list);
            if (reverse) {
                // for arabic rec
                throw new FluxException("Not Supported Yet!!!");
            }
            NDArray conf_list_float64 = conf_list.toType(DataType.FLOAT64, true);
            NDArray conf_list_float64_mean = conf_list_float64.mean();
            double[] scores = conf_list_float64_mean.toDoubleArray();
            result_list.add(new RecognitionResult(text, scores));

            selection.close();
            conf_list_float64_mean.close();
            conf_list_float64.close();
            conf_list.close();
        }
        if (text_prob != null) {
            text_prob.close();
        }
        text_index.close();
        return result_list;
    }

    public List<RecognitionResult> process(NDArray preds) {
        NDArray expandPreds = preds.expandDims(0);
        NDArray preds_idx = expandPreds.argMax(-1);
        NDArray preds_prob = expandPreds.max(new int[] {-1});
        List<RecognitionResult> decodeResults = decode(preds_idx, preds_prob, true);
        preds_prob.close();
        preds_idx.close();
        expandPreds.close();
        return decodeResults;
    }

    public List<RecognitionResult> process(float[][] preds) {
        StringBuilder text = new StringBuilder();
        int previousIndex = -1;
        double scoreSum = 0.0d;
        int scoreCount = 0;

        for (float[] timestep : preds) {
            if (timestep == null || timestep.length == 0) {
                previousIndex = -1;
                continue;
            }

            int maxIndex = 0;
            float maxScore = timestep[0];
            for (int i = 1; i < timestep.length; i++) {
                if (timestep[i] > maxScore) {
                    maxScore = timestep[i];
                    maxIndex = i;
                }
            }

            if (maxIndex != 0 && maxIndex != previousIndex) {
                if (maxIndex < character.size()) {
                    text.append(character.get(maxIndex));
                }
                scoreSum += maxScore;
                scoreCount++;
            }
            previousIndex = maxIndex;
        }

        if (reverse) {
            throw new FluxException("Not Supported Yet!!!");
        }

        double score = scoreCount == 0 ? 0.0d : scoreSum / scoreCount;
        return List.of(new RecognitionResult(text.toString(), new double[]{score}));
    }

    public List<RecognitionResult> process(FloatBuffer preds,
                                           int offset,
                                           int sequenceLength,
                                           int classCount) {
        StringBuilder text = new StringBuilder();
        int previousIndex = -1;
        double scoreSum = 0.0d;
        int scoreCount = 0;

        for (int timestep = 0; timestep < sequenceLength; timestep++) {
            int rowOffset = offset + timestep * classCount;
            int maxIndex = 0;
            float maxScore = preds.get(rowOffset);
            for (int i = 1; i < classCount; i++) {
                float score = preds.get(rowOffset + i);
                if (score > maxScore) {
                    maxScore = score;
                    maxIndex = i;
                }
            }

            if (maxIndex != 0 && maxIndex != previousIndex) {
                if (maxIndex < character.size()) {
                    text.append(character.get(maxIndex));
                }
                scoreSum += maxScore;
                scoreCount++;
            }
            previousIndex = maxIndex;
        }

        if (reverse) {
            throw new FluxException("Not Supported Yet!!!");
        }

        double score = scoreCount == 0 ? 0.0d : scoreSum / scoreCount;
        return List.of(new RecognitionResult(text.toString(), new double[]{score}));
    }

}
