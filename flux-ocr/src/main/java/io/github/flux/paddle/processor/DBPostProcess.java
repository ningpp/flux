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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import clipper2.core.Path64;
import clipper2.core.Paths64;
import clipper2.core.Point64;
import clipper2.offset.ClipperOffset;
import clipper2.offset.EndType;
import clipper2.offset.JoinType;
import io.github.flux.core.MatManager;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * The post process for Differentiable Binarization (DB).
 */
public class DBPostProcess {
    private final float thresh;
    private final float boxThresh;
    private final float unclipRatio;
    private final int maxCandidates;
    private final String scoreMode;
    private final String boxType;
    private final int minSize;
    private static final int MIN_SIZE = 3;

    public DBPostProcess(
            final float thresh,
            final float boxThresh,
            final float unclipRatio,
            final int maxCandidates,
            final String scoreMode,
            final String boxType
    ) {
        // validate scoreMode
        if (!"slow".equals(scoreMode) && !"fast".equals(scoreMode)) {
            throw new IllegalArgumentException(
                    "Score mode must be 'slow' or 'fast' but got: " + scoreMode
            );
        }

        this.thresh = thresh;
        this.boxThresh = boxThresh;
        this.unclipRatio = unclipRatio;
        this.maxCandidates = maxCandidates;
        this.scoreMode = scoreMode;
        this.boxType = boxType;
        this.minSize = MIN_SIZE;
    }

    public float getThresh() {
        return thresh;
    }

    public float getBoxThresh() {
        return boxThresh;
    }

    public float getUnclipRatio() {
        return unclipRatio;
    }

    public static record Point(float x, float y) {

        public Point(float x, float y) {
            this.x = x;
            this.y = y;
        }

        public Point(double x, double y) {
            this((float) x, (float) y);
        }
    }

    private static final int _channels = 2;

    public void fromArray(MatOfPoint2f mp2f, Point[] a) {
        if(a==null || a.length==0)
            return;
        int num = a.length;
        mp2f.alloc(num);
        float[] buff = new float[num * _channels];
        for(int i=0; i<num; i++) {
            Point p = a[i];
            buff[_channels*i] = p.x;
            buff[_channels*i+1] = p.y;
        }
        mp2f.put(0, 0, buff); //TODO: check ret val!
    }

    public void fromArray(MatOfPoint mp, Point[] a) {
        if(a==null || a.length==0)
            return;
        int num = a.length;
        mp.alloc(num);
        int[] buff = new int[num * _channels];
        for(int i=0; i<num; i++) {
            Point p = a[i];
            buff[_channels*i] = (int) p.x;
            buff[_channels*i+1] = (int) p.y;
        }
        mp.put(0, 0, buff); //TODO: check ret val!
    }

    public Point[] toArray(MatOfPoint2f mp) {
        int num = (int) mp.total();
        Point[] ap = new Point[num];
        if(num == 0) {
            return ap;
        }
        float[] buff = new float[num * _channels];
        mp.get(0, 0, buff); //TODO: check ret val!
        for(int i=0; i<num; i++) {
            ap[i] = new Point(buff[i * _channels], buff[i * _channels + 1]);
        }
        return ap;
    }

    /**
     * Apply post-processing to a batch of predictions.
     *
     * @param preds       an NDList whose first element is a batch‐shaped NDArray of raw preds
     * @param imageShapes a List of Shapes, one per image in the batch
     * @param thresh      optional threshold to override this.thresh (nullable)
     * @param boxThresh   optional box threshold to override this.boxThresh (nullable)
     * @param unclipRatio optional unclip ratio to override this.unclipRatio (nullable)
     * @return a Pair of (List of box NDArrays, List of confidence scores)
     */
    public Pair<List<NDArray>, List<Float>> call(
            MatManager matManager,
            final NDList preds,
            final double[] imageShapes,
            final Float thresh,
            final Float boxThresh,
            final Float unclipRatio
    ) {
        List<NDArray> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        // use provided override or fall back to the instance defaults
        float t = (thresh != null ? thresh : this.thresh);
        float bt = (boxThresh != null ? boxThresh : this.boxThresh);
        float ur = (unclipRatio != null ? unclipRatio : this.unclipRatio);

        // preds.get(0) is assumed to be [batch, ...], so split on axis 0
        NDArray batchPred = preds.get(0);
        int size = (int) batchPred.getShape().getShape()[0];
        for (int i = 0; i < size; i++) {
            NDArray pred = batchPred.get(i);

            // process() returns a Pair<NDArray, Float>
            Pair<List<NDArray>, List<Float>> result = this.process(matManager, pred, imageShapes, t, bt, ur);

            boxes.addAll(result.getKey());
            scores.addAll(result.getValue());
            pred.close();
        }

        batchPred.close();
        return Pair.of(boxes, scores);
    }

    /**
     * Process a single prediction into boxes and scores.
     *
     * @param pred        NDArray of shape [1, H, W]
     * @param imageShapes Shape containing [srcH, srcW, ratioH, ratioW] (we only use H and W)
     * @param thresh      binarization threshold
     * @param boxThresh   box confidence threshold
     * @param unclipRatio unclip ratio for box expansion
     * @return a Pair of (List of box NDArrays, List of confidence scores)
     */
    public Pair<List<NDArray>, List<Float>> process(
            MatManager matManager,
            final NDArray pred,
            final double[] imageShapes,
            final float thresh,
            final float boxThresh,
            final float unclipRatio
    ) {
        // 1) segmentation mask: pred > thresh => boolean mask
        NDArray segmentation = pred.gt(thresh);

        // 2) unpack original image size (we ignore ratioH, ratioW here)
        int srcH = Double.valueOf(imageShapes[0]).intValue();
        int srcW = Double.valueOf(imageShapes[1]).intValue();

        // 5) extract boxes & scores based on boxType
        if ("poly".equals(boxType)) {
            Pair<List<NDArray>, List<Float>> result = polygonsFromBitmap(matManager, pred, segmentation, srcW, srcH, boxThresh, unclipRatio);
            segmentation.close();
            pred.close();
            return result;
        } else if ("quad".equals(boxType)) {
            Pair<List<NDArray>, List<Float>> result = boxesFromBitmap(matManager, pred, segmentation, srcW, srcH, boxThresh, unclipRatio);
            segmentation.close();
            pred.close();
            return result;
        } else {
            throw new IllegalArgumentException(
                    "boxType can only be one of ['quad', 'poly'], but got: " + boxType
            );
        }
    }

    public Pair<List<NDArray>, List<Float>> boxesFromBitmap(
            MatManager matManager,
            final NDArray pred,
            final NDArray bitmap,
            final int destWidth,
            final int destHeight,
            final float boxThresh,
            final float unclipRatio
    ) {
        // 1. prepare scales
        long[] bshape = bitmap.getShape().getShape();  // [H, W]
        int height = (int) bshape[0];
        int width = (int) bshape[1];
        float widthScale = (float) destWidth / width;
        float heightScale = (float) destHeight / height;

        List<NDArray> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        // 2. convert bitmap NDArray -> OpenCV Mat (0 or 255)
        /*
        byte[] binBytes = new byte[height * width];
        NDArray bitmapFloat32 = bitmap.toType(DataType.FLOAT32, false);
        float[] flat = bitmapFloat32.toFloatArray();
        for (int i = 0; i < flat.length; i++) {
            binBytes[i] = flat[i] > 0 ? (byte) 255 : (byte) 0;
        }
        Mat mat = matManager.newMat(height, width, CvType.CV_8UC1);
        mat.put(0, 0, binBytes);
        */

        // 2. 将 bitmap * 255 转为 uint8 类型，不发生额外内存复制（in-place）
        NDArray mul = bitmap.mul(255);
        NDArray mask = mul.toType(DataType.UINT8, true);

        // 3. 从 NDArray 中获取底层 byte 数据
        ByteBuffer bb = mask.toByteBuffer();
        byte[] data = new byte[bb.remaining()];
        bb.get(data);
        mask.close();
        mul.close();
        bitmap.close();

        // 4. 构造 OpenCV 的 Mat (单通道 8bit)
        Mat mat = matManager.newMat(height, width, CvType.CV_8UC1);
        mat.put(0, 0, data);

        // 3. find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = matManager.newMat();
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        mat.release();
        hierarchy.release();

        int numContours = Math.min(contours.size(), maxCandidates);
        for (int idx = 0; idx < numContours; idx++) {
            MatOfPoint contour = contours.get(idx);
            Pair<Point[], Float> miniBoxPair = getMiniBoxes(contour);
            Point[] points = miniBoxPair.getKey();
            Float sside = miniBoxPair.getValue();
            if (Float.compare(sside, minSize) < 0) {
                continue;
            }

            // MatOfPoint matOfPoint = new MatOfPoint();
            // fromArray(matOfPoint, points);

            float score = boxScoreFast(pred, points);
            if (score < boxThresh) {
                continue;
            }
            float[][] unclipped = unclip(points, unclipRatio);


            Point[] unclippedPts2f = new Point[unclipped.length];
            for (int i = 0; i < unclipped.length; i++) {
                float x = unclipped[i][0];
                float y = unclipped[i][1];
                unclippedPts2f[i] = new Point(x, y);
            }
            MatOfPoint unclippedMatOfPoint = new MatOfPoint();
            fromArray(unclippedMatOfPoint, unclippedPts2f);

            miniBoxPair = getMiniBoxes(unclippedMatOfPoint);
            unclippedMatOfPoint.release();
            points = miniBoxPair.getKey();
            sside = miniBoxPair.getValue();
            if (Float.compare(sside, minSize + 2) < 0) {
                continue;
            }

            float[][] scaled = new float[points.length][2];
            for (int i = 0; i < points.length; i++) {
                int x = Math.round((float) points[i].x * widthScale);
                int y = Math.round((float) points[i].y * heightScale);
                scaled[i][0] = Math.max(0, Math.min(x, destWidth));
                scaled[i][1] = Math.max(0, Math.min(y, destHeight));
            }

            NDArray boxArr = pred.getManager().create(scaled);
            NDArray boxArrInt32 = boxArr.toType(DataType.INT32, true);
            boxArr.close();
            boxes.add(boxArrInt32);
            scores.add(score);
        }
        return Pair.of(boxes, scores);
    }


    /**
     * Extracts polygonal boxes from a binary bitmap.
     *
     * @param pred        the original prediction NDArray [H, W]
     * @param bitmap      binarized mask NDArray [H, W] (values 0 or 1)
     * @param destWidth   original image width
     * @param destHeight  original image height
     * @param boxThresh   minimum box confidence
     * @param unclipRatio ratio for unclipping polygons
     * @return Pair of (List of box NDArrays, List of box scores)
     */
    public Pair<List<NDArray>, List<Float>> polygonsFromBitmap(
            MatManager matManager,
            final NDArray pred,
            final NDArray bitmap,
            final int destWidth,
            final int destHeight,
            final float boxThresh,
            final float unclipRatio
    ) {
        // 1. prepare scales
        long[] bshape = bitmap.getShape().getShape();  // [H, W]
        int height = (int) bshape[0];
        int width = (int) bshape[1];
        float widthScale = (float) destWidth / width;
        float heightScale = (float) destHeight / height;

        List<NDArray> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        // 2. convert bitmap NDArray -> OpenCV Mat (0 or 255)
        /*
        byte[] binBytes = new byte[height * width];
        NDArray bitmapFloat32 = bitmap.toType(DataType.FLOAT32, false);
        float[] flat = bitmapFloat32.toFloatArray();
        for (int i = 0; i < flat.length; i++) {
            binBytes[i] = flat[i] > 0 ? (byte) 255 : (byte) 0;
        }
        Mat mat = matManager.newMat(height, width, CvType.CV_8UC1);
        mat.put(0, 0, binBytes);
        */

        // 2. 将 bitmap * 255 转为 uint8 类型，不发生额外内存复制（in-place）
        NDArray mul = bitmap.mul(255f);
        NDArray mask = mul.toType(DataType.UINT8, true);
        // saveNDArrayByteToTxt(mask, "d:\\bitmap255ed-java-bytes-1751555076-3c5ffc5292184f76ba44c82cac645bd3.txt", true);

        // 3. 从 NDArray 中获取底层 byte 数据
        ByteBuffer bb = mask.toByteBuffer();
        byte[] data = new byte[bb.remaining()];
        bb.get(data);
        mask.close();
        mul.close();
        bitmap.close();

        // 4. 构造 OpenCV 的 Mat (单通道 8bit)
        Mat mat = matManager.newMat(height, width, CvType.CV_8UC1);
        mat.put(0, 0, data);

        // 3. find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = matManager.newMat();
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        mat.release();
        hierarchy.release();

        // 4. iterate contours up to maxCandidates
        int numContours = Math.min(contours.size(), maxCandidates);
        for (int idx = 0; idx < numContours; idx++) {
            MatOfPoint contour = contours.get(idx);
            // approximate polygon
            MatOfPoint2f pts2f = new MatOfPoint2f(contour.toArray());
            contour.release();
            double epsilon = 0.002 * Imgproc.arcLength(pts2f, true);
            MatOfPoint2f approx2f = new MatOfPoint2f();
            Imgproc.approxPolyDP(pts2f, approx2f, epsilon, true);

            Point[] pts = toArray(approx2f);
            if (pts.length < 4) {
                continue;
            }

            // 5. compute box score
            float score = boxScoreFast(pred, pts);
            if (score < boxThresh) {
                continue;
            }

            // 6. unclip
            float[][] unclipped = unclip(pts, unclipRatio);
            if (unclipped == null || unclipped.length <= 1) {
                continue;
            }

            // Build an array of Point2f
            Point[] unclippedPts2f = new Point[unclipped.length];
            for (int i = 0; i < unclipped.length; i++) {
                float x = unclipped[i][0];
                float y = unclipped[i][1];
                unclippedPts2f[i] = new Point(x, y);
            }
            MatOfPoint unclippedMatOfPoint = new MatOfPoint();
            fromArray(unclippedMatOfPoint, unclippedPts2f);

            // 7. filter by min size
            Pair<Point[], Float> mini = getMiniBoxes(unclippedMatOfPoint);
            Point[] miniPoints = mini.getKey();
            unclippedMatOfPoint.release();
            if (Float.compare(mini.getValue(), minSize + 2) < 0) {
                continue;
            }

            // 8. scale back to original image size
            float[][] scaled = new float[miniPoints.length][2];
            for (int i = 0; i < miniPoints.length; i++) {
                int x = Math.round((float) miniPoints[i].x * widthScale);
                int y = Math.round((float) miniPoints[i].y * heightScale);
                scaled[i][0] = Math.max(0, Math.min(x, destWidth));
                scaled[i][1] = Math.max(0, Math.min(y, destHeight));
            }

            // 9. convert to NDArray and append
            NDArray boxArr = pred.getManager().create(scaled);
            boxes.add(boxArr);
            scores.add(score);
        }

        return Pair.of(boxes, scores);
    }

    /**
     * box_score_fast: use bbox mean score as the mean score
     *
     * @param bitmap NDArray of shape [H, W] with float scores
     * @param boxPts array of Points (x, y) for the polygon vertices
     * @return the mean score inside the polygon
     */
    private float boxScoreFast(
            final NDArray bitmap,
            final Point[] boxPts
    ) {
        // 1) get shape
        long[] shape = bitmap.getShape().getShape();  // [H, W]
        final int h = (int) shape[0];
        final int w = (int) shape[1];

        // 2) compute bounding box of polygon
        double minX = Double.MAX_VALUE, maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE, maxY = Double.MIN_VALUE;
        for (Point p : boxPts) {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        final int xmin = Math.max(0, (int) Math.floor(minX));
        final int xmax = Math.min(w - 1, (int) Math.ceil(maxX));
        final int ymin = Math.max(0, (int) Math.floor(minY));
        final int ymax = Math.min(h - 1, (int) Math.ceil(maxY));

        // 3) create mask for the region
        final int maskH = ymax - ymin + 1;
        final int maskW = xmax - xmin + 1;
        Mat mask = Mat.zeros(maskH, maskW, CvType.CV_8UC1);

        // 4) shift polygon coords and fill
        Point[] shifted = new Point[boxPts.length];
        for (int i = 0; i < boxPts.length; i++) {
            shifted[i] = new Point(boxPts[i].x - xmin, boxPts[i].y - ymin);
        }
        MatOfPoint mop = new MatOfPoint();
        fromArray(mop, shifted);
        List<MatOfPoint> ptsList = new ArrayList<>();
        ptsList.add(mop);
        Imgproc.fillPoly(mask, ptsList, new Scalar(1));

        // 5) flatten bitmap to Java array for fast access
        float[] flat = bitmap.toType(DataType.FLOAT32, false).toFloatArray();

        // 6) read mask bytes and compute mean over polygon
        byte[] maskBytes = new byte[maskH * maskW];
        mask.get(0, 0, maskBytes);
        double sum = 0;
        int count = 0;
        for (int yy = ymin; yy <= ymax; yy++) {
            for (int xx = xmin; xx <= xmax; xx++) {
                int mi = (yy - ymin) * maskW + (xx - xmin);
                if ((maskBytes[mi] & 0xFF) != 0) {
                    sum += flat[yy * w + xx];
                    count++;
                }
            }
        }
        return count > 0 ? (float) (sum / count) : 0f;
    }


    public void points(RotatedRect minRect, Point[] pt)
    {
        double _angle = minRect.angle * Math.PI / 180.0;
        double b = (double) Math.cos(_angle) * 0.5f;
        double a = (double) Math.sin(_angle) * 0.5f;

        pt[0] = new Point(
                minRect.center.x - a * minRect.size.height - b * minRect.size.width,
                minRect.center.y + b * minRect.size.height - a * minRect.size.width);

        pt[1] = new Point(
                minRect.center.x + a * minRect.size.height - b * minRect.size.width,
                minRect.center.y - b * minRect.size.height - a * minRect.size.width);

        pt[2] = new Point(
                2 * minRect.center.x - pt[0].x,
                2 * minRect.center.y - pt[0].y);

        pt[3] = new Point(
                2 * minRect.center.x - pt[1].x,
                2 * minRect.center.y - pt[1].y);
    }

    /**
     * get mini boxes: finds the minimum-area bounding box and orders its 4 points
     *
     * @param contour a MatOfPoint or MatOfPoint2f representing the polygon contour
     * @return Pair of (array of 4 ordered Points, length of the shorter side of the box)
     */
    private Pair<Point[], Float> getMiniBoxes(final MatOfPoint contour) {
        // 1) convert to MatOfPoint2f if needed
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        // 2) get the minimum-area rotated rectangle
        RotatedRect minRect = Imgproc.minAreaRect(contour2f);

        // 3) extract its 4 corner points
        Point[] pts = new Point[4];
        points(minRect, pts);  // fills pts[]

        // 4) sort by x-coordinate
        Arrays.sort(pts, Comparator.comparingDouble(p -> p.x));

        // 5) decide top/bottom ordering for left and right pairs
        int idx1, idx4, idx2, idx3;
        // leftmost two are pts[0], pts[1]
        if (pts[1].y > pts[0].y) {
            idx1 = 0;
            idx4 = 1;
        } else {
            idx1 = 1;
            idx4 = 0;
        }
        // rightmost two are pts[2], pts[3]
        if (pts[3].y > pts[2].y) {
            idx2 = 2;
            idx3 = 3;
        } else {
            idx2 = 3;
            idx3 = 2;
        }

        // 6) build the ordered box: [top-left, top-right, bottom-right, bottom-left]
        Point[] box = new Point[]{
                pts[idx1],
                pts[idx2],
                pts[idx3],
                pts[idx4]
        };

        // 7) compute the length of the shorter side of the rectangle
        float shortSide = (float) Math.min(minRect.size.width, minRect.size.height);

        contour2f.release();
        contour.release();

        return Pair.of(box, shortSide);
    }


    /**
     * Unclip (offset) a polygon using Clipper2-java with Path64.
     *
     * @param boxPts      array of Points (x, y) defining the polygon contour
     * @param unclipRatio ratio by which to offset (distance = area * ratio / perimeter)
     * @return expanded polygon as a 2D array [N][2] of (x, y) coordinates
     */
    private float[][] unclip(
            final Point[] boxPts,
            final float unclipRatio
    ) {
        // 1) 计算面积和周长
        MatOfPoint2f contour2f = new MatOfPoint2f();
        fromArray(contour2f, boxPts);
        double area = Imgproc.contourArea(contour2f);
        double length = Imgproc.arcLength(contour2f, true);
        double distance = area * unclipRatio / length;

        // 2) 构造 Path64 并添加顶点（四舍五入到 long）
        Path64 subject = new Path64();
        for (Point p : boxPts) {
            subject.add(new Point64(Math.round(p.x), Math.round(p.y)));
        }

        // 3) 执行偏移：AddPath + Execute
        ClipperOffset offsetter = new ClipperOffset();
        offsetter.AddPath(subject, JoinType.Round, EndType.Polygon);

        Paths64 solution = new Paths64();
        offsetter.Execute(distance, solution);

        // 4) 取第一个结果路径并转换回 float[][]
        if (solution.isEmpty()) {
            return new float[0][];
        }
        Path64 expandedPath = solution.get(0);
        float[][] expanded = new float[expandedPath.size()][2];
        for (int i = 0; i < expandedPath.size(); i++) {
            Point64 pt = expandedPath.get(i);
            expanded[i][0] = pt.x;
            expanded[i][1] = pt.y;
        }
        return expanded;
    }

}
