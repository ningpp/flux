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

import io.github.flux.core.TopkResult;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Select top-k predictions with float precision.
 */
public class TopkProcessor {

    private final Map<Integer, String> labelIdMap;

    public TopkProcessor(List<String> labels) {
        this.labelIdMap = labels == null ? Collections.emptyMap() :
                Collections.unmodifiableMap(IntStream.range(0, labels.size())
                        .boxed().collect(Collectors.toMap(i -> i, labels::get)));
    }

    public TopkResult compute(float[][] preds, int k) {
        int n = preds.length;
        var indices = new int[n][k];
        var scores = new float[n][k];
        var labels = new String[n][k];

        for (var i = 0; i < n; i++) {
            var row = preds[i];
            var idx = IntStream.range(0, row.length)
                    .boxed()
                    .sorted((a, b) -> Double.compare(row[b], row[a]))
                    .limit(k)
                    .toArray(Integer[]::new);
            for (var j = 0; j < k; j++) {
                indices[i][j] = idx[j];
                scores[i][j] = row[idx[j]];
                labels[i][j] = labelIdMap.getOrDefault(idx[j], String.valueOf(idx[j]));
            }
        }
        return new TopkResult(indices, scores, labels);
    }

}
