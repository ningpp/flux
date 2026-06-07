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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.exception.FluxException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Post-processor for PP-DocLayoutV3 model outputs.
 * <p>
 * Handles three ONNX outputs: logits, pred_boxes, and order_logits.
 * The reading order is computed using a voting-based ranking algorithm
 * (from HuggingFace transformers' {@code _get_order_seqs}):
 * <ol>
 *   <li>Sigmoid on order_logits → order_scores (N×N pairwise probability matrix)</li>
 *   <li>Upper-triangular sum of order_scores + lower-triangular sum of (1 - order_scores^T)
 *       → order_votes (how many "come after" votes each query receives)</li>
 *   <li>Argsort on order_votes → order_pointers (sorted query indices)</li>
 *   <li>Scatter ranks → order_seq (rank/position for each query)</li>
 * </ol>
 * Detection results are sorted by order_seq before returning.
 */
public class PPDocLayoutV3PostProcessor {
    private final List<String> labelNames;

    public PPDocLayoutV3PostProcessor(List<String> labelNames) {
        this.labelNames = labelNames;
    }

    /**
     * Post-processes PP-DocLayoutV3 outputs into final bounding boxes sorted by reading order.
     *
     * @param outLogits    classification logits, shape (batch, numQueries, numClasses)
     * @param outBbox      predicted boxes in cxcywh format, shape (batch, numQueries, 4)
     * @param orderLogits  reading order logits, shape (batch, numQueries, numQueries)
     * @param threshold    score threshold to keep predictions
     * @param targetSizes  [2][batch] where [0] = heights, [1] = widths of original images
     * @return detection results per image, sorted by reading order
     */
    public List<List<ObjectDetectionResult>> process(NDArray outLogits, NDArray outBbox,
                                                      NDArray orderLogits, float threshold,
                                                      int[][] targetSizes) {
        NDManager manager = outLogits.getManager();

        // 1. Compute reading order sequences via voting-based ranking
        int[] orderSeqs = getOrderSeqs(orderLogits);

        // 2. Convert boxes from cxcywh to xyxy (normalized)
        NDArray boxes = centerToCornersFormat(outBbox);

        // 3. Scale boxes to target sizes
        if (targetSizes != null) {
            NDArray imgH = manager.create(targetSizes[0]);
            NDArray imgW = manager.create(targetSizes[1]);
            NDArray scaleFct = NDArrays.stack(new NDList(imgW, imgH, imgW, imgH), 1).expandDims(1);
            boxes = boxes.mul(scaleFct);
        }

        // 4. Sigmoid on logits → scores, then topk
        NDArray scores = outLogits.getNDArrayInternal().sigmoid();
        int numTopQueries = (int) outLogits.getShape().get(1);
        int numClasses = (int) outLogits.getShape().get(2);

        NDArray flattenedScores = scores.flatten(1, -1);
        NDList topKResult = flattenedScores.topK(numTopQueries, -1, true, true);
        NDArray topScores = topKResult.get(0);
        NDArray indices = topKResult.get(1);
        flattenedScores.close();

        // 5. Derive labels and box (query) indices
        NDArray labels = indices.mod(numClasses);
        NDArray boxIndices = indices.div(numClasses).floor();

        // 6. Gather boxes using box indices
        long[] boxShape = boxes.getShape().getShape();
        NDArray expandedBoxIndices = boxIndices.expandDims(-1).tile(new long[]{1, 1, boxShape[boxShape.length - 1]});
        NDArray finalBoxes = boxes.gather(expandedBoxIndices, 1);

        // 7. Gather order sequences using box indices
        int batchSize = (int) topScores.getShape().get(0);
        int topK = (int) topScores.getShape().get(1);
        int numQueries = (int) orderLogits.getShape().get(1);
        int[] gatheredOrderSeqs = new int[batchSize * topK];
        for (int b = 0; b < batchSize; b++) {
            int[] batchBoxIndices = boxIndices.get(b).toType(DataType.INT32, true).toIntArray();
            for (int k = 0; k < topK; k++) {
                int qi = Math.max(0, Math.min(batchBoxIndices[k], numQueries - 1));
                gatheredOrderSeqs[b * topK + k] = orderSeqs[b * numQueries + qi];
            }
        }

        // 8. Process each image in the batch
        List<List<ObjectDetectionResult>> allResults = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            NDArray imageScores = topScores.get(i);
            NDArray imageLabels = labels.get(i);
            NDArray imageBoxes = finalBoxes.get(i);

            // Filter by threshold (use >= to match Python's score >= threshold)
            NDArray mask = imageScores.gte(threshold);
            NDArray filteredScores = imageScores.booleanMask(mask);
            NDArray filteredLabels = imageLabels.booleanMask(mask);
            var boxMask = mask.expandDims(-1).tile(new long[]{1, boxShape[boxShape.length - 1]});
            NDArray filteredBoxes = imageBoxes.booleanMask(boxMask, 0);

            // Gather order sequences for filtered detections
            float[] imgScores = filteredScores.toFloatArray();
            float[] imgBboxes = filteredBoxes.expandDims(-1).toFloatArray();
            int[] imgLabels = filteredLabels.toType(DataType.INT32, true).toIntArray();

            // Build (index, orderSeq) pairs for filtered detections
            int size = imgScores.length;
            Integer[] filterIndices = new Integer[size];
            int[] imgOrderSeqs = new int[size];
            int filterIdx = 0;
            for (int k = 0; k < topK; k++) {
                if (imageScores.getFloat(k) > threshold) {
                    filterIndices[filterIdx] = filterIdx;
                    imgOrderSeqs[filterIdx] = gatheredOrderSeqs[i * topK + k];
                    filterIdx++;
                }
            }

            // Sort by order sequence (reading order)
            Integer[] sortIndices = new Integer[size];
            for (int j = 0; j < size; j++) {
                sortIndices[j] = j;
            }
            Arrays.sort(sortIndices, Comparator.comparingInt(j -> imgOrderSeqs[j]));

            List<ObjectDetectionResult> results = new ArrayList<>(size);
            for (int idx : sortIndices) {
                float[] coordinate = new float[4];
                System.arraycopy(imgBboxes, idx * 4, coordinate, 0, 4);
                int clsId = imgLabels[idx];
                String label = clsId < labelNames.size() ? labelNames.get(clsId) : String.valueOf(clsId);
                results.add(new ObjectDetectionResult(clsId, label, imgScores[idx], coordinate));
            }

            allResults.add(results);

            // Clean up per-image NDArrays
            filteredScores.close();
            filteredLabels.close();
            filteredBoxes.close();
        }

        // Clean up intermediate NDArrays
        boxes.close();
        topScores.close();
        indices.close();
        labels.close();
        boxIndices.close();
        expandedBoxIndices.close();
        finalBoxes.close();
        scores.close();

        return allResults;
    }

    /**
     * Computes reading order sequences using the voting-based ranking algorithm.
     * <p>
     * This matches the HuggingFace transformers' {@code _get_order_seqs} method:
     * <ol>
     *   <li>sigmoid on order_logits → order_scores (N×N pairwise)</li>
     *   <li>order_votes[j] = sum_{i&lt;j} order_scores[i,j] + sum_{i&gt;j} (1 - order_scores[j,i])
     *       (i.e., triu(order_scores, diag=1).sum(dim=1) + tril(1 - order_scores^T, diag=-1).sum(dim=1),
     *        where sum(dim=1) sums over rows per column)</li>
     *   <li>argsort(order_votes) → order_pointers</li>
     *   <li>scatter ranks → order_seq</li>
     * </ol>
     *
     * @param orderLogits shape (batch, numQueries, numQueries)
     * @return flat int array of order sequences, layout [batch][query]
     */
    private int[] getOrderSeqs(NDArray orderLogits) {
        NDArray orderScores = orderLogits.getNDArrayInternal().sigmoid();  // (batch, N, N)

        int batchSize = (int) orderScores.getShape().get(0);
        int seqLen = (int) orderScores.getShape().get(1);

        float[] scoresArr = orderScores.toFloatArray();
        orderScores.close();

        int[] orderSeqs = new int[batchSize * seqLen];

        for (int b = 0; b < batchSize; b++) {
            int base = b * seqLen * seqLen;

            // Compute order_votes per column j (matching PyTorch's sum(dim=1)):
            // order_votes[j] = sum_{i<j} order_scores[i,j] + sum_{i>j} (1 - order_scores[j,i])
            float[] votes = new float[seqLen];
            for (int j = 0; j < seqLen; j++) {
                float triuSum = 0;  // sum_{i<j} order_scores[i,j]: i comes before j
                float trilSum = 0;  // sum_{i>j} (1 - order_scores[j,i]): i comes after j
                for (int i = 0; i < seqLen; i++) {
                    if (i < j) {
                        triuSum += scoresArr[base + i * seqLen + j];
                    }
                    if (i > j) {
                        trilSum += (1.0f - scoresArr[base + j * seqLen + i]);
                    }
                }
                votes[j] = triuSum + trilSum;
            }

            // argsort by votes ascending → order_pointers
            Integer[] pointers = new Integer[seqLen];
            for (int i = 0; i < seqLen; i++) pointers[i] = i;
            Arrays.sort(pointers, Comparator.comparingDouble(i -> votes[i]));

            // Scatter ranks: order_seq[pointers[k]] = k
            for (int k = 0; k < seqLen; k++) {
                orderSeqs[b * seqLen + pointers[k]] = k;
            }
        }

        return orderSeqs;
    }

    /**
     * Converts center format bounding boxes to corners format.
     */
    private static NDArray centerToCornersFormat(NDArray bboxesCenter) {
        NDArray centerX = bboxesCenter.get("..., 0");
        NDArray centerY = bboxesCenter.get("..., 1");
        NDArray width = bboxesCenter.get("..., 2");
        NDArray height = bboxesCenter.get("..., 3");

        NDArray halfWidth = width.mul(0.5);
        NDArray halfHeight = height.mul(0.5);

        NDArray topLeftX = centerX.sub(halfWidth);
        NDArray topLeftY = centerY.sub(halfHeight);
        NDArray bottomRightX = centerX.add(halfWidth);
        NDArray bottomRightY = centerY.add(halfHeight);

        return NDArrays.stack(new NDList(topLeftX, topLeftY, bottomRightX, bottomRightY), -1);
    }
}
