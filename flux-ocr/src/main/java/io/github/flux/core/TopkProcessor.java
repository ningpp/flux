package io.github.flux.core;

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
