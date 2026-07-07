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
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Save Result Transform
 *
 * <p>This class is responsible for post-processing detection results, including thresholding,
 * non-maximum suppression (NMS), and restructuring the boxes based on the input type (normal or
 * rotated object detection).
 */
public class DetPostProcessor {

    private static final Logger LOGGER = LoggerFactory.getLogger(DetPostProcessor.class);

    private final List<String> labels;

    /**
     * Initializes the DetPostProcess class.
     *
     * @param labels The list of labels for the detection categories.
     */
    public DetPostProcessor(List<String> labels) {
        this.labels = labels;
    }

    /**
     * Applies post-processing to the detection boxes.
     *
     * @param boxes                 The input detection boxes with scores.
     * @param imgSize               The original image size as a long array [height, width].
     * @param threshold             The threshold to apply to the detection scores. Can be a Float or a Map<Integer, Float>.
     * @param layoutNms             Whether to apply layout non-maximum suppression.
     * @param layoutUnclipRatio     The ratio to unclip the boxes. Can be a Float, a float array of size 2, or a Map.
     * @param layoutMergeBboxesMode The mode for merging bounding boxes. Can be a String or a Map.
     * @return The post-processed detection boxes as an NDList.
     */
    public List<ObjectDetectionResult> process(
            NDArray boxes,
            long[] imgSize,
            Object threshold,
            boolean layoutNms,
            Object layoutUnclipRatio,
            Object layoutMergeBboxesMode) {
        NDManager manager = boxes.getManager();
        NDArray currentBoxes = boxes.duplicate();

        // Apply threshold
        if (threshold instanceof Number) {
            float thresh = ((Number) threshold).floatValue();
            NDArray scores = currentBoxes.get(":, 1");
            NDArray categories = currentBoxes.get(":, 0");
            NDArray mask = scores.gt(thresh).logicalAnd(categories.gt(-1));
            NDArray filtered = currentBoxes.get(mask);
            currentBoxes.close();
            currentBoxes = filtered;
        } else if (threshold instanceof Map) {
            Map<Integer, Float> thresholdMap = (Map<Integer, Float>) threshold;
            List<NDArray> filteredBoxesList = new ArrayList<>();
            NDArray uniqueCategories = manager.create(Arrays.stream(currentBoxes.get(":, 0").toIntArray()).distinct().toArray());

            for (NDArray catIdArray : uniqueCategories.split(uniqueCategories.getShape().get(0))) {
                int catId = catIdArray.toIntArray()[0];
                NDArray categoryMask = currentBoxes.get(":, 0").eq(catId);
                NDArray categoryBoxes = currentBoxes.get(categoryMask);

                float categoryThreshold = thresholdMap.getOrDefault(catId, 0.5f);
                NDArray scores = categoryBoxes.get(":, 1");
                NDArray categories = categoryBoxes.get(":, 0");
                NDArray mask = scores.gt(categoryThreshold).logicalAnd(categories.gt(-1));
                filteredBoxesList.add(categoryBoxes.get(mask));
            }
            NDArray filtered;
            if (!filteredBoxesList.isEmpty()) {
                filtered = NDArrays.concat(new NDList(filteredBoxesList), 0);
            } else {
                filtered = manager.create(new Shape(0, currentBoxes.getShape().get(1)));
            }
            currentBoxes.close();
            currentBoxes = filtered;
        }

        // Apply layout NMS
        if (layoutNms) {
            List<Integer> selected_indices = nms(currentBoxes, 0.6f, 0.98f);
            // First, convert the List<Integer> to a long[] array.
            // DJL indexing often uses long arrays.
            long[] indices = selected_indices.stream().mapToLong(i -> i).toArray();
            // Create a 1D NDArray from the indices array.
            NDArray selectedIndices = manager.create(indices);
            NDArray nmsBoxes = currentBoxes.get(selectedIndices);
            currentBoxes.close();
            currentBoxes = nmsBoxes;
        }

        // Filter large boxes for "image" category
        boolean filterLargeImage = true;
        if (filterLargeImage && currentBoxes.getShape().get(0) > 1 && currentBoxes.getShape().get(1) == 6) {
            float areaThres = (imgSize[0] > imgSize[1]) ? 0.82f : 0.93f;
            int imageIndex = labels.indexOf("image");
            if (imageIndex != -1) {
                long imgArea = imgSize[0] * imgSize[1];
                List<NDArray> filteredBoxes = new ArrayList<>();
                for (NDArray box : currentBoxes.split(currentBoxes.getShape().get(0))) {
                    NDArray boxSqueezed = box.squeeze(0);
                    int labelIndex = (int) boxSqueezed.getFloat(0);
                    if (labelIndex == imageIndex) {
                        float xmin = Math.max(0, boxSqueezed.getFloat(2));
                        float ymin = Math.max(0, boxSqueezed.getFloat(3));
                        float xmax = Math.min(imgSize[0], boxSqueezed.getFloat(4));
                        float ymax = Math.min(imgSize[1], boxSqueezed.getFloat(5));
                        float boxArea = (xmax - xmin) * (ymax - ymin);
                        if (boxArea <= areaThres * imgArea) {
                            filteredBoxes.add(boxSqueezed.expandDims(0));
                        }
                    } else {
                        filteredBoxes.add(boxSqueezed.expandDims(0));
                    }
                }
                if (!filteredBoxes.isEmpty()) {
                    NDArray concatBoxes = NDArrays.concat(new NDList(filteredBoxes), 0);
                    currentBoxes.close();
                    currentBoxes = concatBoxes;
                }
            }
        }

        // Merge bounding boxes based on mode
        if (layoutMergeBboxesMode != null) {
            Integer formulaIndex = labels.contains("formula") ? labels.indexOf("formula") : null;
            if (layoutMergeBboxesMode instanceof String) {
                String mode = (String) layoutMergeBboxesMode;
                if (!Arrays.asList("union", "large", "small").contains(mode)) {
                    throw new IllegalArgumentException("layout_merge_bboxes_mode must be one of ['union', 'large', 'small']");
                }
                if (!"union".equals(mode)) {
                    NDArray[] containment = checkContainment(currentBoxes, formulaIndex, -1, mode); // Pass -1 for category_index
                    NDArray containedByOther = containment[1];
                    NDArray filtered;
                    if ("large".equals(mode)) {
                        filtered = currentBoxes.get(containedByOther.eq(0));
                    } else { // small
                        NDArray containsOther = containment[0];
                        filtered = currentBoxes.get(containsOther.eq(0).logicalOr(containedByOther.eq(1)));
                    }
                    currentBoxes.close();
                    currentBoxes = filtered;
                }
            } else if (layoutMergeBboxesMode instanceof Map) {
                Map<Integer, String> modeMap = (Map<Integer, String>) layoutMergeBboxesMode;
                NDArray keepMask = manager.ones(new Shape(currentBoxes.getShape().get(0)), DataType.BOOLEAN);

                for (Map.Entry<Integer, String> entry : modeMap.entrySet()) {
                    Integer categoryIndex = entry.getKey();
                    String mode = entry.getValue();
                    if (!"union".equals(mode)) {
                        NDArray[] containment = checkContainment(currentBoxes, formulaIndex, categoryIndex, mode);
                        NDArray containsOther = containment[0];
                        NDArray containedByOther = containment[1];
                        if ("large".equals(mode)) {
                            keepMask = keepMask.logicalAnd(containedByOther.eq(0));
                        } else { // small
                            keepMask = keepMask.logicalAnd(containsOther.eq(0).logicalOr(containedByOther.eq(1)));
                        }
                    }
                }
                NDArray mergedBoxes = currentBoxes.get(keepMask);
                currentBoxes.close();
                currentBoxes = mergedBoxes;
            }
        }


        if (currentBoxes.size() == 0) {
            return List.of();
        }

        if (currentBoxes.getShape().get(1) == 8) {
            currentBoxes = sortBoxes(manager, currentBoxes);
        }
        /*
        if boxes.shape[1] == 8:
            # Sort boxes by their order
            sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
         */

        // Unclip boxes
        if (layoutUnclipRatio != null) {
            currentBoxes = unclipBoxes(currentBoxes, layoutUnclipRatio);
        }

        // Restructure boxes based on shape
        long boxShape = currentBoxes.getShape().get(1);
        if (boxShape == 6) {
            // For Normal Object Detection
            return restructuredBoxes(currentBoxes, labels, imgSize);
        } else if (boxShape == 10) {
            // Adapt For Rotated Object Detection
            return restructuredRotatedBoxes(currentBoxes, labels, imgSize);
        } else {
            throw new FluxException(
                    "The shape of boxes should be 6 or 10, instead of " + boxShape);
        }
    }

    /*
     # Sort boxes by their order
     */
    private static NDArray sortBoxes(NDManager manager, NDArray boxes) {
        Shape shape = boxes.getShape();
        if (shape.get(1) != 8) {
            return boxes;
        }

        int n = (int) shape.get(0);

        // 转成 Java 数组
        float[][] boxArr = ArrayUtil.convertToFloatArray(boxes);

        // 构造索引数组
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // 等价于 np.lexsort((-boxes[:,7], boxes[:,6]))
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                // 先按第 6 列升序
                int cmp = Float.compare(boxArr[i1][6], boxArr[i2][6]);
                if (cmp != 0) {
                    return cmp;
                }
                // 再按第 7 列降序
                return Float.compare(boxArr[i2][7], boxArr[i1][7]);
            }
        });

        // 排序并裁剪到前 6 列
        float[][] sorted = new float[n][6];
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            System.arraycopy(boxArr[idx], 0, sorted[i], 0, 6);
        }

        boxes.close();
        // 转回 NDArray
        return manager.create(sorted);
    }

    /**
     * Calculates the Intersection over Union (IoU) of two bounding boxes.
     *
     * @param box1 A 1D NDArray representing the coordinates [xmin, ymin, xmax, ymax] of the first box.
     * @param box2 A 1D NDArray representing the coordinates [xmin, ymin, xmax, ymax] of the second box.
     * @return The IoU value as a float.
     */
    public static float iou(NDArray box1, NDArray box2) {
        float xmin1 = box1.getFloat(0);
        float ymin1 = box1.getFloat(1);
        float xmax1 = box1.getFloat(2);
        float ymax1 = box1.getFloat(3);

        float xmin2 = box2.getFloat(0);
        float ymin2 = box2.getFloat(1);
        float xmax2 = box2.getFloat(2);
        float ymax2 = box2.getFloat(3);

        // Calculate the coordinates of the intersection rectangle
        float interXmin = Math.max(xmin1, xmin2);
        float interYmin = Math.max(ymin1, ymin2);
        float interXmax = Math.min(xmax1, xmax2);
        float interYmax = Math.min(ymax1, ymax2);

        // Calculate the area of intersection
        float interArea = Math.max(0, interXmax - interXmin) * Math.max(0, interYmax - interYmin);

        // Calculate the area of both bounding boxes
        float box1Area = (xmax1 - xmin1) * (ymax1 - ymin1);
        float box2Area = (xmax2 - xmin2) * (ymax2 - ymin2);

        // Calculate the IoU
        float unionArea = box1Area + box2Area - interArea;

        if (unionArea <= 0) {
            return 0.0f;
        }

        return interArea / unionArea;
    }

    /**
     * Performs Non-Maximum Suppression (NMS) with different IoU thresholds for same and different classes.
     *
     * @param boxes A 2D NDArray of shape [N, 6] where each row is [cls_id, score, xmin, ymin, xmax, ymax].
     * @param iouSame The IoU threshold for boxes of the same class.
     * @param iouDiff The IoU threshold for boxes of different classes.
     * @return A list of indices of the selected boxes.
     */
    public static List<Integer> nms(NDArray boxes, float iouSame, float iouDiff) {
        // Extract scores and get the number of boxes
        NDArray scores = boxes.get(":, 1");
        long numBoxes = boxes.getShape().get(0);

        // Generate a list of indices [0, 1, 2, ..., N-1]
        List<Integer> indices = IntStream.range(0, (int) numBoxes).boxed().collect(Collectors.toList());

        // Sort indices based on scores in descending order
        indices.sort(Comparator.comparingDouble((Integer i) -> scores.getFloat(i)).reversed());

        List<Integer> selectedIndices = new ArrayList<>();

        while (!indices.isEmpty()) {
            // Get the index of the box with the highest score
            int currentIdx = indices.get(0);
            selectedIndices.add(currentIdx);

            // Get the properties of the current box
            NDArray currentBox = boxes.get(currentIdx);
            int currentClass = (int) currentBox.getFloat(0);
            NDArray currentCoords = currentBox.get("2:");

            // Remove the current index from the list
            indices.remove(0);

            // Prepare a list to hold indices that are kept for the next iteration
            List<Integer> filteredIndices = new ArrayList<>();
            for (int idx : indices) {
                NDArray box = boxes.get(idx);
                int boxClass = (int) box.getFloat(0);
                NDArray boxCoords = box.get("2:");

                float iouValue = iou(currentCoords, boxCoords);
                float threshold = (currentClass == boxClass) ? iouSame : iouDiff;

                // If the IoU is below the threshold, keep the box
                if (iouValue < threshold) {
                    filteredIndices.add(idx);
                }
            }
            // Update the list of indices for the next iteration
            indices = filteredIndices;
        }

        return selectedIndices;
    }

    /**
     * Placeholder for checking containment between boxes.
     *
     * @param boxes         NDArray of boxes
     * @param formulaIndex  Index of the 'formula' label
     * @param categoryIndex Index of the category to check, or -1 for all
     * @param mode          The mode ("large" or "small")
     * @return An array of two NDArrays: [contains_other, contained_by_other]
     */
    private NDArray[] checkContainment(NDArray boxes, Integer formulaIndex, int categoryIndex, String mode) {
        LOGGER.warn("Warning: checkContainment() is a placeholder and needs to be implemented.");
        // TODO: Implement your box containment logic here.
        // This should return two boolean NDArrays indicating which boxes contain others
        // and which are contained by others.
        // Use the parent manager (not a sub-manager) so the returned arrays remain valid after return.
        NDManager manager = boxes.getManager();
        long numBoxes = boxes.getShape().get(0);
        return new NDArray[]{
                manager.zeros(new Shape(numBoxes), DataType.INT32),
                manager.zeros(new Shape(numBoxes), DataType.INT32)
        };
    }

    /**
     * Placeholder for unclipping (expanding) boxes.
     *
     * @param boxes             NDArray of boxes
     * @param layoutUnclipRatio Ratio for unclipping. Can be Float, float[], or Map.
     * @return Expanded boxes
     */
    private NDArray unclipBoxes(NDArray boxes, Object layoutUnclipRatio) {
        LOGGER.warn("Warning: unclipBoxes() is a placeholder and needs to be implemented.");
        // TODO: Implement your unclip logic here.
        return boxes;
    }

    /**
     * Placeholder for restructuring normal detection boxes.
     * Restructures the given bounding boxes and labels based on the image size.
     *
     * @param boxes    A 2D NDArray of bounding boxes(shape: [N, 6]), where each box is represented as
     *                 [cls_id, score, xmin, ymin, xmax, ymax].
     * @param labels   A list of class labels corresponding to the class ids.
     * @param imgSize  A long array representing the width and height of the image, i.e., [width, height].
     * @return A list of BoundingBox objects, each containing cls_id, label, score, and coordinate.
     */
    public List<ObjectDetectionResult> restructuredBoxes(NDArray boxes, List<String> labels, long[] imgSize) {
        List<ObjectDetectionResult> boxList = new ArrayList<>();
        long w = imgSize[0];
        long h = imgSize[1];

        // Iterate over each box in the NDArray
        for (NDArray box : boxes.split(boxes.getShape().get(0))) {
            // Squeeze to remove the dimension of size 1, making it a 1D array
            NDArray boxData = box.squeeze(0);

            float[] coords = boxData.get("2:").toFloatArray();
            float xmin = coords[0];
            float ymin = coords[1];
            float xmax = coords[2];
            float ymax = coords[3];

            // Clip coordinates to image boundaries
            xmin = Math.max(0, xmin);
            ymin = Math.max(0, ymin);
            xmax = Math.min(w, xmax);
            ymax = Math.min(h, ymax);

            // Skip invalid boxes where the area is zero or negative
            if (xmax <= xmin || ymax <= ymin) {
                continue;
            }

            int clsId = (int) boxData.getFloat(0);
            float score = boxData.getFloat(1);
            String label = labels.get(clsId);
            float[] finalCoordinate = new float[]{xmin, ymin, xmax, ymax};

            boxList.add(new ObjectDetectionResult(clsId, label, score, finalCoordinate));
        }

        return boxList;
    }

    /**
     * Placeholder for restructuring rotated detection boxes.
     *
     * @param boxes   A 2D array of rotated bounding boxes with each box represented as [cls_id, score, x1, y1, x2, y2, x3, y3, x4, y4].
     * @param labels  A list of class labels corresponding to the class ids.
     * @param imgSize A tuple representing the width and height of the image.
     * @return Restructured boxes, likely as an NDList of different results.
     */
    private List<ObjectDetectionResult> restructuredRotatedBoxes(NDArray boxes, List<String> labels, long[] imgSize) {
        // Assert that the shape of the boxes is [N, 10]
        if (boxes.getShape().get(1) != 10) {
            throw new IllegalArgumentException(
                    "The shape of rotated boxes should be [N, 10], but got " + boxes.getShape());
        }

        List<ObjectDetectionResult> boxList = new ArrayList<>();
        long w = imgSize[0];
        long h = imgSize[1];

        // Iterate over each box in the NDArray
        for (NDArray box : boxes.split(boxes.getShape().get(0))) {
            // Squeeze to remove the dimension of size 1, making it a 1D array
            NDArray boxData = box.squeeze(0);

            // Get all 8 coordinate points from index 2 onwards
            float[] points = boxData.get("2:").toFloatArray();

            // Clip each point to the image boundaries
            float x1 = Math.min(Math.max(0, points[0]), w);
            float y1 = Math.min(Math.max(0, points[1]), h);
            float x2 = Math.min(Math.max(0, points[2]), w);
            float y2 = Math.min(Math.max(0, points[3]), h);
            float x3 = Math.min(Math.max(0, points[4]), w);
            float y3 = Math.min(Math.max(0, points[5]), h);
            float x4 = Math.min(Math.max(0, points[6]), w);
            float y4 = Math.min(Math.max(0, points[7]), h);

            int clsId = boxData.getInt(0);
            float score = boxData.getFloat(1);
            String label = labels.get(clsId);
            float[] finalCoordinate = new float[]{x1, y1, x2, y2, x3, y3, x4, y4};

            boxList.add(new ObjectDetectionResult(clsId, label, score, finalCoordinate));
        }

        return boxList;
    }
}