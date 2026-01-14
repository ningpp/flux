package io.github.flux.util;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Comparator;

public final class ImageUtil {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    private ImageUtil() {
    }

    public static Mat crop(MatManager matManager, Mat src, int startx, int starty, int endx, int endy) {
        // 定义矩形区域
        Rect roi = new Rect(startx, starty, endx - startx, endy - starty);
        Mat mat = matManager.newMat(src, roi);

        Mat result = mat.clone();
        IOUtil.close(mat);
        return result;
    }

    /**
     * Get the minimum area rectangle crop from the given image and points.
     *
     * @param manager the DJL NDManager
     * @param mat   the input image as an NDArray (H×W×3 RGB)
     * @param points  an N×2 int array of points defining the shape to crop
     * @return the cropped image as an NDArray
     */
    public static Mat getMinAreaRectCrop(MatManager matManager, NDManager manager, Mat mat, int[][] points) {
        if (points.length != 4 || points[0].length != 2) {
            throw new IllegalArgumentException("points must be an N×2 array with N>=3");
        }

        // 2) Build MatOfPoint2f from int[][] points
        Point[] srcPts = new Point[points.length];
        for (int i = 0; i < points.length; i++) {
            srcPts[i] = new Point(points[i][0], points[i][1]);
        }
        MatOfPoint2f srcMat = new MatOfPoint2f(srcPts);

        // 3) Compute min-area rotated rect
        RotatedRect rect = Imgproc.minAreaRect(srcMat);
        srcMat.release();

        // 4) Extract its 4 corner points
        Point[] boxPts = new Point[4];
        rect.points(boxPts);

        // 5) Sort by x (ascending)
        Arrays.sort(boxPts, Comparator.comparingDouble(p -> p.x));

        // 6) Identify top‐&‐bottom among left and right pairs
        Point left1 = boxPts[0], left2 = boxPts[1];
        Point right1 = boxPts[2], right2 = boxPts[3];

        Point tl = (left1.y < left2.y) ? left1 : left2;
        Point bl = (left1.y < left2.y) ? left2 : left1;
        Point tr = (right1.y < right2.y) ? right1 : right2;
        Point br = (right1.y < right2.y) ? right2 : right1;

        // 7) Build the 4‐point box in order: [tl, tr, br, bl]
        int[][] boxInt = new int[4][2];
        Point[] ordered = new Point[]{tl, tr, br, bl};
        for (int i = 0; i < 4; i++) {
            boxInt[i][0] = (int) Math.round(ordered[i].x);
            boxInt[i][1] = (int) Math.round(ordered[i].y);
        }

        // 8) Delegate to your rotate‐crop routine
        return getRotateCropImage(matManager, mat, boxInt);
    }

    public static Mat getRotateCropImage(MatManager matManager, Mat mat, int[][] points) {
        if (points.length != 4 || points[0].length != 2) {
            throw new IllegalArgumentException("points must be 4×2");
        }

        // Build Point[] from int[][], casting to double
        Point[] srcPts = new Point[4];
        for (int i = 0; i < 4; i++) {
            srcPts[i] = new Point(points[i][0], points[i][1]);
        }

        // 3) Compute crop width & height via Math.hypot
        double wA = Math.hypot(srcPts[0].x - srcPts[1].x, srcPts[0].y - srcPts[1].y);
        double wB = Math.hypot(srcPts[2].x - srcPts[3].x, srcPts[2].y - srcPts[3].y);
        int cropW = (int) Math.max(wA, wB);

        double hA = Math.hypot(srcPts[0].x - srcPts[3].x, srcPts[0].y - srcPts[3].y);
        double hB = Math.hypot(srcPts[1].x - srcPts[2].x, srcPts[1].y - srcPts[2].y);
        int cropH = (int) Math.max(hA, hB);

        // Perspective transform
        MatOfPoint2f srcMat = new MatOfPoint2f(srcPts);
        MatOfPoint2f dstMat = new MatOfPoint2f(
                new Point(0, 0),
                new Point(cropW, 0),
                new Point(cropW, cropH),
                new Point(0, cropH)
        );
        Mat M = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        Mat warped = matManager.newMat();
        Imgproc.warpPerspective(
                mat, warped, M,
                new Size(cropW, cropH),
                Imgproc.INTER_CUBIC,
                Core.BORDER_REPLICATE
        );

        srcMat.release();
        dstMat.release();
        M.release();

        // Rotate if tall
        if ((double) warped.rows() / warped.cols() >= 1.5) {
            return rotate90Degree(matManager, warped);
        }

        return warped;
    }

    // Method using OpenCV
    public static Mat rotate90Degree(MatManager matManager, Mat srcImg) {
        Mat dstImg = matManager.newMat();
        // Rotate 90 degrees counterclockwise (equivalent to np.rot90)
        Core.rotate(srcImg, dstImg, Core.ROTATE_90_COUNTERCLOCKWISE);
        srcImg.release();
        return dstImg;
    }

    public static Mat bgrToRgb(MatManager matManager, Mat bgrImg) {
        Mat rgbImg = matManager.newMat();
        Imgproc.cvtColor(bgrImg, rgbImg, Imgproc.COLOR_BGR2RGB);
        bgrImg.release();
        return rgbImg;
    }

    public static Mat readToRgb(MatManager matManager, String image) {
        Mat bgrImg = Imgcodecs.imread(image, Imgcodecs.IMREAD_COLOR_BGR);
        Mat rgbImg = matManager.newMat();
        Imgproc.cvtColor(bgrImg, rgbImg, Imgproc.COLOR_BGR2RGB);
        bgrImg.release();
        return rgbImg;
    }

    public static Mat rotateImage(MatManager matManager, Mat image, double angle) {
        if (angle < 0 || angle >= 360) {
            throw new IllegalArgumentException("`angle` should be in range [0, 360)");
        }

        if (angle < 1e-7) {
            return image;
        }

        // Get image dimensions
        int h = image.rows();
        int w = image.cols();

        // Calculate center point
        Point center = new Point(w / 2.0, h / 2.0);
        double scale = 1.0;

        // Get rotation matrix
        Mat mat = Imgproc.getRotationMatrix2D(center, angle, scale);

        // Extract cos and sin values from rotation matrix
        double cos = Math.abs(mat.get(0, 0)[0]);
        double sin = Math.abs(mat.get(0, 1)[0]);

        // Calculate new dimensions
        int newW = (int) ((h * sin) + (w * cos));
        int newH = (int) ((h * cos) + (w * sin));

        // Adjust translation components
        double[] translation02 = mat.get(0, 2);
        double[] translation12 = mat.get(1, 2);
        translation02[0] += (newW - w) / 2.0;
        translation12[0] += (newH - h) / 2.0;
        mat.put(0, 2, translation02);
        mat.put(1, 2, translation12);

        // Create destination size
        Size dstSize = new Size(newW, newH);

        // Perform the rotation
        Mat rotated = matManager.newMat();
        Imgproc.warpAffine(
                image,
                rotated,
                mat,
                dstSize,
                Imgproc.INTER_CUBIC
        );

        IOUtil.close(mat);
        return rotated;
    }

    /**
     * Converts an NDArray of shape eg: [3, 3, 32, 300] (RGB) to grayscale using the given weights.
     * The function applies: 0.2989 * R + 0.5870 * G + 0.1140 * B along axis 1.
     *
     * @param input NDArray of shape eg: [3, 3, 32, 300], channel-first (N, C, H, W)
     * @return NDArray of shape eg: [3, 1, 32, 300], single grayscale channel
     */
    public static NDArray rgbToGray(NDArray input) {
        // Select R, G, B channels: (axis 1)
        NDArray r = input.get(":,0:1,:,:");
        NDArray g = input.get(":,1:2,:,:");
        NDArray b = input.get(":,2:3,:,:");

        // Apply weights and sum
        NDArray gray = r.mul(0.2989)
                .add(g.mul(0.5870))
                .add(b.mul(0.1140));
        return gray;
    }



    public static BufferedImage matToBufferedImage(Mat mat) {
        int type;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else if (mat.channels() == 3) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        } else {
            throw new IllegalArgumentException("Unsupported number of channels: " + mat.channels());
        }

        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer); // 将像素数据拷贝到buffer

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }

    public static Mat bufferedImageToMat(MatManager matManager, BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        Mat mat;

        switch (image.getType()) {
            case BufferedImage.TYPE_BYTE_GRAY:
                mat = matManager.newMat(height, width, CvType.CV_8UC1);
                byte[] grayData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
                mat.put(0, 0, grayData);
                break;

            case BufferedImage.TYPE_3BYTE_BGR:
                mat = matManager.newMat(height, width, CvType.CV_8UC3);
                byte[] bgrData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
                mat.put(0, 0, bgrData);
                break;

            case BufferedImage.TYPE_4BYTE_ABGR:
                mat = matManager.newMat(height, width, CvType.CV_8UC4);
                byte[] abgrData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
                mat.put(0, 0, abgrData);
                break;

            default:
                // 如果类型不支持，转换为 TYPE_3BYTE_BGR 处理
                BufferedImage converted = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
                Graphics2D g = converted.createGraphics();
                g.drawImage(image, 0, 0, null);
                g.dispose();
                return bufferedImageToMat(matManager, converted);
        }

        return mat;
    }

    public static Mat normalize(MatManager matManager, Mat image, Scalar mean, Scalar std) {
        Mat normalizedImage = matManager.newMat();

        // Subtract mean: (image - mean)
        Mat meanSubtracted = matManager.newMat();
        Core.subtract(image, mean, meanSubtracted);

        // Divide by std: (image - mean) / std
        Core.divide(meanSubtracted, std, normalizedImage);

        image.release();
        meanSubtracted.release();
        return normalizedImage;
    }

    /**
     * Alternative method with custom padding color
     */
    // TODO 这不太对
    public static Mat padImageToSize(MatManager matManager, Mat image, int targetWidth, int targetHeight, Scalar paddingColor) {
        int currentWidth = image.cols();
        int currentHeight = image.rows();

        int paddingRight = Math.max(0, targetWidth - currentWidth);
        int paddingBottom = Math.max(0, targetHeight - currentHeight);
        if (paddingRight == 0 && paddingBottom == 0) {
            return image;
        }

        Mat paddedImage = matManager.newMat();

        Core.copyMakeBorder(
                image,
                paddedImage,
                0,               // top
                paddingBottom,   // bottom
                0,               // left
                paddingRight,    // right
                Core.BORDER_CONSTANT,
                paddingColor
        );

        image.release();
        return paddedImage;
    }

    /**
     * Adds a border to the image. This is an equivalent of Pillow's ImageOps.expand.
     *
     * @param image  The image to expand, as an OpenCV Mat.
     * @param border The border width in pixels.
     * @param fill   The pixel fill value (a color value). Default is 0 (black).
     * @return A new image with the specified border.
     */
    public static Mat expand(MatManager matManager, Mat image, int border, Scalar fill) {
        return expand(matManager, image, border, border, border, border, fill);
    }

    /**
     * Adds a border to the image.
     *
     * @param image      The image to expand, as an OpenCV Mat.
     * @param border     A tuple of 2 integers for horizontal and vertical borders (left/right, top/bottom).
     * @param fill       The pixel fill value (a color value). Default is 0 (black).
     * @return A new image with the specified border.
     */
    public static Mat expand(MatManager matManager, Mat image, int[] border, Scalar fill) {
        if (border.length != 2) {
            throw new IllegalArgumentException("Border array must have 2 elements for horizontal and vertical padding.");
        }
        int horizontal = border[0];
        int vertical = border[1];
        return expand(matManager, image, vertical, horizontal, vertical, horizontal, fill);
    }

    /**
     * Adds a border to the image.
     *
     * @param image      The image to expand, as an OpenCV Mat.
     * @param border     A tuple of 4 integers for border widths (top, bottom, left, right).
     * @param fill       The pixel fill value (a color value). Default is 0 (black).
     * @return A new image with the specified border.
     */
    public static Mat expand(MatManager matManager, Mat image, int[] border, Scalar fill, boolean isLRTB) {
        if (border.length != 4) {
            throw new IllegalArgumentException("Border array must have 4 elements for top, bottom, left, right padding.");
        }
        // The python version uses (left, top, right, bottom)
        // OpenCV's copyMakeBorder uses (top, bottom, left, right)
        if (isLRTB) {
            return expand(matManager, image, border[1], border[3], border[0], border[2], fill);
        } else {
            return expand(matManager, image, border[0], border[1], border[2], border[3], fill);
        }
    }

    /**
     * Internal helper to add a border to the image using OpenCV's copyMakeBorder.
     *
     * @param image  The source image.
     * @param top    Padding width for the top.
     * @param bottom Padding width for the bottom.
     * @param left   Padding width for the left.
     * @param right  Padding width for the right.
     * @param fill   The color of the border.
     * @return A new Mat object with the added border.
     */
    private static Mat expand(MatManager matManager, Mat image, int top, int bottom, int left, int right, Scalar fill) {
        Mat dest = matManager.newMat();
        Core.copyMakeBorder(image, dest, top, bottom, left, right, Core.BORDER_CONSTANT, fill);
        return dest;
    }

    public static NDArray toNDArrayUint8(Mat mat, NDManager manager) {
        byte[] buf = new byte[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(ByteBuffer.wrap(buf), shape, DataType.UINT8);
    }

    public static NDArray toNDArrayFloat(Mat mat, NDManager manager) {
        float[] buf = new float[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(FloatBuffer.wrap(buf), shape, DataType.FLOAT32);
    }

    public static NDArray toChannalNDArrayFloat(MatManager matManager, Mat mat, NDManager manager) {
        float[] buf = new float[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.channels(), mat.height(), mat.width());
        return manager.create(FloatBuffer.wrap(buf), shape, DataType.FLOAT32);
    }

    public static NDArray toNDArrayFloat(MatManager matManager, Mat image, NDManager manager, Image.Flag flag) {
        Mat mat = matManager.newMat();
        if (flag == Image.Flag.GRAYSCALE) {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2GRAY);
        } else {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2RGB);
        }
        float[] buf = new float[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(FloatBuffer.wrap(buf), shape, DataType.FLOAT32);
    }

    public static NDArray toNDArray(MatManager matManager, Mat image, NDManager manager, Image.Flag flag) {
        Mat mat = matManager.newMat();
        if (flag == Image.Flag.GRAYSCALE) {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2GRAY);
        } else {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2RGB);
        }
        byte[] buf = new byte[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(ByteBuffer.wrap(buf), shape, DataType.UINT8);
    }

}
