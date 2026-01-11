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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class SortQuadBoxes implements SortBoxes {

    /**
     * Sort quad boxes in order from top→bottom, left→right.
     *
     * @param manager DJL NDManager
     * @param dtPolys NDArray of shape [N,4,2], dtype can be anything but will be cast to int32
     * @return sorted boxes as NDArray [N,4,2], dtype int32
     */
    public NDArray sort(NDManager manager, NDArray dtPolys) {
        // 1) Ensure int32 and get raw int[]
        NDArray intPolys = dtPolys.toType(DataType.INT32, true);
        Shape s = intPolys.getShape();       // [N,4,2]
        long numBoxes = s.get(0);
        int total = (int) (numBoxes * 4 * 2);
        int[] flat = intPolys.toIntArray();

        // 2) Pull into Java list of int[4][2]
        int stride = 4 * 2;
        List<int[][]> boxes = new ArrayList<>((int) numBoxes);
        for (int i = 0; i < numBoxes; i++) {
            int[][] quad = new int[4][2];
            int base = i * stride;
            for (int p = 0; p < 4; p++) {
                quad[p][0] = flat[base + p * 2];
                quad[p][1] = flat[base + p * 2 + 1];
            }
            boxes.add(quad);
        }

        // 3) Initial sort by (y of first point, then x of first point)
        boxes.sort(Comparator
                .comparingInt((int[][] box) -> box[0][1])
                .thenComparingInt(box -> box[0][0])
        );

        // 4) Intra‑band insertion sort
        for (int i = 0; i < boxes.size() - 1; i++) {
            for (int j = i; j >= 0; j--) {
                int y1 = boxes.get(j)[0][1];
                int y2 = boxes.get(j + 1)[0][1];
                int x1 = boxes.get(j)[0][0];
                int x2 = boxes.get(j + 1)[0][0];
                if (Math.abs(y2 - y1) < 10 && x2 < x1) {
                    Collections.swap(boxes, j, j + 1);
                } else {
                    break;
                }
            }
        }

        // 5) Flatten back into an int[]
        int[] outFlat = new int[total];
        for (int i = 0; i < boxes.size(); i++) {
            int[][] quad = boxes.get(i);
            int base = i * stride;
            for (int p = 0; p < 4; p++) {
                outFlat[base + p * 2]     = quad[p][0];
                outFlat[base + p * 2 + 1] = quad[p][1];
            }
        }

        // 6) Return a new INT32 NDArray of shape [N,4,2]
        intPolys.close();
        return manager.create(outFlat, new Shape(numBoxes, 4, 2));
    }

}
