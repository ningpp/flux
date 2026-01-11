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
import java.util.List;

public class RtdetrPostProcessor {
    private final List<String> labelNames;

    public RtdetrPostProcessor(List<String> labelNames) {
        this.labelNames = labelNames;
    }


    /**
     * Converts center format bounding boxes to corners format
     *
     * @param bboxesCenter NDArray with shape [..., 4] containing [center_x, center_y, width, height]
     * @return NDArray with shape [..., 4] containing [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
     */
    private static NDArray centerToCornersFormat(NDArray bboxesCenter) {
        NDManager manager = bboxesCenter.getManager();

        // Split the last dimension to get center_x, center_y, width, height
        NDArray centerX = bboxesCenter.get("..., 0");
        NDArray centerY = bboxesCenter.get("..., 1");
        NDArray width = bboxesCenter.get("..., 2");
        NDArray height = bboxesCenter.get("..., 3");

        // Calculate half width and height
        NDArray halfWidth = width.mul(0.5);
        NDArray halfHeight = height.mul(0.5);

        // Calculate corners: top_left_x, top_left_y, bottom_right_x, bottom_right_y
        NDArray topLeftX = centerX.sub(halfWidth);
        NDArray topLeftY = centerY.sub(halfHeight);
        NDArray bottomRightX = centerX.add(halfWidth);
        NDArray bottomRightY = centerY.add(halfHeight);

        // Stack along the last dimension
        return NDArrays.stack(new NDList(topLeftX, topLeftY, bottomRightX, bottomRightY), -1);
    }

    /**
     * Post-processes object detection outputs into final bounding boxes
     *
     * @param outLogits
     * @param outBbox
     * @param threshold    Score threshold to keep predictions (default: 0.5)
     * @param targetSizes  [height, width]
     * @param useFocalLoss Whether focal loss was used during training (default: true)
     * @return
     */
    public List<List<ObjectDetectionResult>> process(NDArray outLogits, NDArray outBbox, float threshold,
                                                     int[][] targetSizes, boolean useFocalLoss) {

        NDManager manager = outLogits.getManager();

        // Convert from relative cxcywh to absolute xyxy
        NDArray boxes = centerToCornersFormat(outBbox);

        // Scale boxes if target sizes are provided
        if (targetSizes != null) {
            // Extract image height and width
            NDArray imgH = manager.create(targetSizes[0]);
            NDArray imgW = manager.create(targetSizes[1]);

            // Create scale factor [img_w, img_h, img_w, img_h]
            NDArray scaleFct = NDArrays.stack(new NDList(imgW, imgH, imgW, imgH), 1);

            // Expand dimensions for broadcasting: [batch_size, 1, 4]
            scaleFct = scaleFct.expandDims(1);

            // Scale boxes
            boxes = boxes.mul(scaleFct);
        }

        int numTopQueries = (int) outLogits.getShape().get(1);
        int numClasses = (int) outLogits.getShape().get(2);

        NDArray scores;
        NDArray labels;
        NDArray finalBoxes;

        if (useFocalLoss) {
            // Apply sigmoid activation
            scores = outLogits.getNDArrayInternal().sigmoid();

            // Flatten and get top k
            // python code is scores.flatten(1)
            NDArray flattenedScores = scores.flatten(1, -1);
            NDList topKResult = flattenedScores.topK(numTopQueries, -1, true, true);
            scores = topKResult.get(0);
            NDArray indices = topKResult.get(1);

            // Calculate labels and box indices
            labels = indices.mod(numClasses);
            // TODO
            // python code is index = index // num_classes
            NDArray boxIndices = indices.div(numClasses).floor();

            // Gather boxes using indices
            long[] boxShape = boxes.getShape().getShape();
            // boxIndices = boxIndices.expandDims(-1).repeat(-1, boxShape[boxShape.length-1]);
            boxIndices = boxIndices.expandDims(-1).tile(new long[] { 1, 1, boxShape[boxShape.length-1]});
            finalBoxes = boxes.gather(boxIndices, 1);

        } else {
            throw new FluxException("Not Support Yet!!!");
        }

        // Process results for each image in the batch
        List<List<ObjectDetectionResult>> allResults = new ArrayList<>();
        int batchSize = (int) scores.getShape().get(0);

        long[] boxShape = boxes.getShape().getShape();
        for (int i = 0; i < batchSize; i++) {
            NDArray imageScores = scores.get(i);
            NDArray imageLabels = labels.get(i);
            NDArray imageBoxes = finalBoxes.get(i);

            // Create mask for scores above threshold
            NDArray mask = imageScores.gt(threshold);

            // Filter based on threshold
            NDArray filteredScores = imageScores.booleanMask(mask);
            NDArray filteredLabels = imageLabels.booleanMask(mask);
            var x = mask.expandDims(-1).tile(new long[] {1, boxShape[boxShape.length-1]});
            NDArray filteredBoxes = imageBoxes.booleanMask(x, 0);

            List<ObjectDetectionResult> results = new ArrayList<>();
            float[] resultScores = filteredScores.toFloatArray();
            float[] resultBboxes = filteredBoxes.expandDims(-1).toFloatArray();
            int[] resultLabels = filteredLabels.toType(DataType.INT32, true).toIntArray();
            int size = resultScores.length;
            for (int j = 0; j < size; j++) {
                float[] coordinate = new float[4];
                System.arraycopy(resultBboxes, j * 4, coordinate, 0, 4);
                results.add(new ObjectDetectionResult(resultLabels[j], labelNames.get(resultLabels[j]), resultScores[j], coordinate));
            }
            allResults.add(results);
        }

        return allResults;
    }

}
