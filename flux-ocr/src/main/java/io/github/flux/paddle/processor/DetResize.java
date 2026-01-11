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

import ai.djl.modality.cv.Image;
import io.github.flux.exception.FluxException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DetResize {

    public record DetResizeResultV2(Mat resizeImg, double[] imgShape) {
    }

    private final int resizeType;
    private final int maxSideLimit;
    private final int limitSideLen;
    private final Integer resizeLong;
    private final LimitType limitType;
    private final int[] imageShape;
    protected final Image.Interpolation interp;

    public DetResize(int resizeType, int limitSideLen, LimitType limitType) {
        this(resizeType, limitSideLen, limitType, null, 4000, null, null);
    }

    public DetResize(int resizeType, int limitSideLen, LimitType limitType,
                     Integer resizeLong, int maxSideLimit, int[] imageShape,
                     Image.Interpolation interp) {
        this.limitSideLen = limitSideLen;
        this.resizeLong = resizeLong;
        this.limitType = limitType;
        this.maxSideLimit = maxSideLimit;
        this.interp = interp;
        if (imageShape != null && imageShape.length > 0) {
            this.imageShape = imageShape;
            this.resizeType = 1;
        } else {
            this.imageShape = null;
            this.resizeType = resizeType;
        }
    }

    public List<DetResizeResultV2> process(List<Mat> imgs) {
        return process(imgs, null, null, null);
    }

    public List<DetResizeResultV2> process(List<Mat> imgs, Integer limitSideLen, LimitType limitType, Integer maxSideLimit) {
        var res = new ArrayList<DetResizeResultV2>();
        for (var img : imgs) {
            var resized = detResize(img, limitSideLen, limitType, maxSideLimit);
            res.add(resized);
            if (img != resized.resizeImg()) {
                img.release();
            }
        }
        return res;
    }

    private DetResizeResultV2 detResize(Mat img, Integer limitSideLen, LimitType limitType, Integer maxSideLimit) {
        int msLimit = maxSideLimit == null ? this.maxSideLimit : maxSideLimit;
        return resize(img, limitSideLen, limitType, msLimit);
    }

    private DetResizeResultV2 resize(Mat ximg, Integer limitSideLen, LimitType limitType, int maxSideLimit) {
        int srcH = ximg.rows();
        int srcW = ximg.cols();
        Mat img;
        if (srcH + srcW < 64) {
            img = imagePadding(ximg, 0);
        } else {
            img = ximg;
        }

        DetResizeResultV2 result;
        if (resizeType == 0) {
            result = resizeImageType0(img, limitSideLen, limitType, maxSideLimit);
        } else if (resizeType == 2) {
            result = resizeImageType2(img, resizeLong);
        } else if (resizeType == 3) {
            result = resizeImageType3(img, new int[0]);
        } else {
            result = resizeImageType1(img);
        }
        if (ximg != img) {
            ximg.release();
            img.release();
        }
        return new DetResizeResultV2(result.resizeImg(),
                new double[]{srcH, srcW, result.imgShape()[0], result.imgShape()[1]});
    }

    public DetResizeResultV2 resizeImageType0(Mat img, Integer requestLimitSideLen, LimitType limitType, int maxSideLimit) {
        // img shape [h, w, c]
        int h = img.rows();
        int w = img.cols();

        double ratio = calculateResizeRatio(h, w, requestLimitSideLen, limitType);
        int resizeH = (int) Math.round(h * ratio);
        int resizeW = (int) Math.round(w * ratio);

        // Apply max side limit if configured
        int maxResizedSide = Math.max(resizeH, resizeW);
        if (maxResizedSide > maxSideLimit) {
            ratio = (double) maxSideLimit / maxResizedSide;
            resizeH = (int) Math.round(resizeH * ratio);
            resizeW = (int) Math.round(resizeW * ratio);
        }

        // Ensure dimensions are multiples of 32 and at least 32
        resizeH = Math.max((int) Math.round(resizeH / 32.0) * 32, 32);
        resizeW = Math.max((int) Math.round(resizeW / 32.0) * 32, 32);

        // Check if no resize needed
        if (resizeH == h && resizeW == w) {
            return new DetResizeResultV2(img, new double[]{1.0, 1.0});
        }

        // Calculate aspect ratios
        double ratioH = (double) resizeH / h;
        double ratioW = (double) resizeW / w;

        Image.Interpolation interpolation = interp == null ? Image.Interpolation.BILINEAR : interp;
        Mat resizedImg = new Resize(resizeW, resizeH, interpolation).process(img);
        img.release();

        return new DetResizeResultV2(resizedImg, new double[]{ratioH, ratioW});
    }

    private double calculateResizeRatio(int h, int w, Integer requestLimitSideLen, LimitType requestLimitType) {
        int _limitSideLen = requestLimitSideLen != null ? requestLimitSideLen : this.limitSideLen;

        double ratio = 1.0;
        int maxSide = Math.max(h, w);
        int minSide = Math.min(h, w);

        LimitType _limitType = requestLimitType != null ? requestLimitType : this.limitType;
        switch (_limitType) {
            case MAX:
                if (maxSide > _limitSideLen) {
                    ratio = (double) _limitSideLen / maxSide;
                }
                break;
            case MIN:
                if (minSide < _limitSideLen) {
                    ratio = (double) _limitSideLen / minSide;
                }
                break;
            case RESIZE_LONG:
                ratio = (double) _limitSideLen / maxSide;
                break;
            default:
                throw new IllegalArgumentException("Unsupported limit type: " + _limitType);
        }
        return ratio;
    }

    public DetResizeResultV2 resizeImageType1(Mat img) {
        if (imageShape == null || imageShape.length < 2) {
            throw new FluxException("illeage imageShape: " + Arrays.toString(imageShape));
        }
        int resizeH = imageShape[0];
        int resizeW = imageShape[1];

        int oriH = img.rows();
        int oriW = img.cols();

        if (resizeH == oriH && resizeW == oriW) {
            return new DetResizeResultV2(img, new double[]{1.0, 1.0});
        }

        double ratioH = (double) resizeH / oriH;
        double ratioW = (double) resizeW / oriW;

        Mat resizedImg = new Resize(resizeW, resizeH, Image.Interpolation.BILINEAR).process(img);
        img.release();

        return new DetResizeResultV2(resizedImg, new double[]{ratioH, ratioW});
    }

    private static final int MAX_STRIDE = 128;

    public static DetResizeResultV2 resizeImageType2(Mat img, int resizeLong) {
        // inputShape: (h, w, c)
        int h = img.rows();
        int w = img.cols();

        int resizeW = w;
        int resizeH = h;

        // Calculate resize ratio based on longer side
        if (resizeH > resizeW) {
            double ratio = (double) resizeLong / resizeH;
            resizeH = (int) (h * ratio);
            resizeW = (int) (w * ratio);
        } else {
            double ratio = (double) resizeLong / resizeW;
            resizeH = (int) (h * ratio);
            resizeW = (int) (w * ratio);
        }

        // Adjust dimensions to be multiples of MAX_STRIDE
        resizeH = (resizeH + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE;
        resizeW = (resizeW + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE;

        // Check if no resize needed
        if (resizeH == h && resizeW == w) {
            return new DetResizeResultV2(img, new double[]{1.0, 1.0});
        }

        Mat resizedImg = new Resize(resizeW, resizeH, Image.Interpolation.BILINEAR).process(img);
        img.release();

        // Calculate aspect ratios
        double ratioH = (double) resizeH / h;
        double ratioW = (double) resizeW / w;
        return new DetResizeResultV2(resizedImg, new double[]{ratioH, ratioW});
    }


    public static DetResizeResultV2 resizeImageType3(Mat img, int[] inputShape) {
        // inputShape: (c, h, w)
        int resizeH = inputShape[1];
        int resizeW = inputShape[2];

        int oriH = img.rows();
        int oriW = img.cols();

        if (resizeH == oriH && resizeW == oriW) {
            return new DetResizeResultV2(img, new double[]{1.0, 1.0});
        }

        double ratioH = (double) resizeH / oriH;
        double ratioW = (double) resizeW / oriW;

        Mat resizedImg = new Resize(resizeW, resizeH, Image.Interpolation.BILINEAR).process(img);
        img.release();
        return new DetResizeResultV2(resizedImg, new double[]{ratioH, ratioW});
    }

    public Mat imagePadding(Mat im, int value) {
        int h = im.rows();
        int w = im.cols();
        int maxH = Math.max(32, h);
        int maxW = Math.max(32, w);

        // Create a matrix of ones with the same type as the input image
        Mat ones = Mat.ones(maxH, maxW, im.type());
        Mat imPad = new Mat();
        // Multiply by the scalar value to set all elements to the desired value
        Core.multiply(ones, new Scalar(value), imPad);

        // Define the ROI and copy the original image into it
        Mat roi = new Mat(imPad, new Rect(0, 0, w, h));
        im.copyTo(roi);

        ones.release();
        roi.release();
        im.release();
        return imPad;
    }

}
