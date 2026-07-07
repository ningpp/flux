package io.github.flux.util;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ImageUtil {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    private ImageUtil() {
    }

    public static OnnxTensor matToOnnxTensor(List<Mat> transformedResults,
                                             final OrtEnvironment env) throws OrtException {
        int height = transformedResults.get(0).rows();
        int width = transformedResults.get(0).cols();
        int channels = transformedResults.get(0).channels();
        int oneSize = (int) (transformedResults.get(0).total() * channels);
        int size = transformedResults.size() * oneSize;
        float[] floatDatas = new float[size];
        // 复用单块临时 buffer，避免每帧重新分配 7.68MB，降低 GC 压力与峰值内存
        float[] oneDatas = new float[oneSize];
        int index = 0;
        for (Mat pad : transformedResults) {
            pad.get(0, 0, oneDatas);
            System.arraycopy(oneDatas, 0, floatDatas, index, oneDatas.length);
            index += oneSize;
        }

        FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
        long[] shape = new long[] {
                transformedResults.size(),
                channels,
                height,
                width
        };
        return OnnxTensor.createTensor(env, dataBuffer, shape);
    }

    public static List<Mat> padImageToSame(MatManager matManager, List<Mat> images) {
        int maxWidth = 0;
        int maxHeight = 0;

        // Load images and find maximum dimensions
        for (Mat image : images) {
            maxWidth = Math.max(maxWidth, image.cols());
            maxHeight = Math.max(maxHeight, image.rows());
        }

        Scalar paddingColor = new Scalar(255, 255, 255);
        List<Mat> results = new ArrayList<>(images.size());
        for (Mat image : images) {
            Mat pad = ImageUtil.padImageToSize(matManager, image, maxWidth, maxHeight, paddingColor);
            results.add(pad);
        }
        return results;
    }

    public static Mat toCHWOfByte(MatManager matManager, Mat img) {

        // Split channels
        List<Mat> channels = matManager.split(img);

        int height = img.rows();
        int width = img.cols();
        int channelSize = height * width;

        byte[] chwData = new byte[3 * channelSize];

        for (int c = 0; c < 3; c++) {
            Mat channel = channels.get(c);
            byte[] channelData = new byte[channelSize];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, chwData, c * channelSize, channelSize);
            IOUtil.close(channel);
        }

        Mat chw = matManager.newMat(height, width, CvType.CV_8SC3);
        chw.put(0, 0, chwData);

        return chw;
    }

    public static Mat toCHWOfFloat(MatManager matManager, Mat img) {

        // Split channels
        List<Mat> channels = matManager.split(img);

        int height = img.rows();
        int width = img.cols();
        int channelSize = height * width;

        float[] chwData = new float[3 * channelSize];

        for (int c = 0; c < 3; c++) {
            Mat channel = channels.get(c);
            float[] channelData = new float[channelSize];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, chwData, c * channelSize, channelSize);
            IOUtil.close(channel);
        }

        Mat chw = matManager.newMat(height, width, CvType.CV_32FC3);
        chw.put(0, 0, chwData);

        return chw;
    }

    /**
     * 将灰度 Mat（H x W）转换为 [1, H, W] 数组
     * @param gray 灰度图像，CV_8U或CV_32F, shape (H, W)
     * @return float[1][H][W]，可直接用于ch为首的神经网络输入
     */
    public static Mat toOneChannelCHW(MatManager matManager, Mat gray) {
        int h = gray.rows();
        int w = gray.cols();
        float[][][] chw = new float[1][h][w];
        float[] flat = new float[h * w];
        // 确保数据为 float32，先转为 float
        Mat grayFloat = matManager.newMat();
        if (gray.type() != CvType.CV_32F) {
            gray.convertTo(grayFloat, CvType.CV_32F);
        } else {
            grayFloat = gray;
        }
        grayFloat.get(0, 0, flat); // 拉成一维
        // 还原 (H, W)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                chw[0][i][j] = flat[i * w + j];
            }
        }
        Mat chwMat = matManager.newMat(h, w, CvType.CV_32FC1);
        chwMat.put(0, 0, ArrayUtil.flat(chw));
        return chwMat;
    }

    public static Mat padding(MatManager matManager, Mat img, int requiredSize) {
        int h = img.rows();
        int w = img.cols();
        int padRight = requiredSize - w;
        int padBottom = requiredSize - h;
        if (padRight < 0 || padBottom < 0) {
            throw new IllegalArgumentException("Image larger than requiredSize!");
        }

        Mat padded = matManager.newMat();
        Core.copyMakeBorder(img, padded,
                0, padBottom,   // top, bottom
                0, padRight,    // left, right
                Core.BORDER_CONSTANT,
                Scalar.all(0)   // 填充为0
        );
        return padded;
    }

    /**
     * 剪裁图片外部的均色（边角最多出现的BGR色）边框
     *
     * @param image 输入BGR三通道图像，CV_8UC3
     * @return 被裁剪后的图像（仍为BGR三通道）
     */
    public static Mat trimWhiteBorder(MatManager matManager, Mat image) {
        // 检查输入
        if (image.channels() != 3 || image.dims() != 2) {
            throw new IllegalArgumentException("Image is not in BGR (3-channel, HxW) format");
        }
        if (image.type() != CvType.CV_8UC3) {
            throw new IllegalArgumentException("Image should be CV_8UC3 (uint8)");
        }

        int rows = image.rows();
        int cols = image.cols();

        // 读取四个角的颜色，并找最多的
        List<Scalar> corners = Arrays.asList(
                new Scalar(image.get(0, 0)),
                new Scalar(image.get(0, cols - 1)),
                new Scalar(image.get(rows - 1, 0)),
                new Scalar(image.get(rows - 1, cols - 1))
        );
        Scalar bgColor = mostCommonColor(corners);

        // 填充背景图 bg
        Mat bg = matManager.newMat(image.size(), image.type(), bgColor);

        // 差异掩码
        Mat diff = matManager.newMat();
        Core.absdiff(image, bg, diff);
        Mat grayDiff = matManager.newMat();
        Imgproc.cvtColor(diff, grayDiff, Imgproc.COLOR_BGR2GRAY);

        // 二值化
        int threshold = 15;
        Mat mask = matManager.newMat();
        Imgproc.threshold(grayDiff, mask, threshold, 255, Imgproc.THRESH_BINARY);

        // 计算包围矩形
        Mat nonZero = matManager.newMat();
        Core.findNonZero(mask, nonZero); // 得到所有非零点
        if (nonZero.empty()) {
            // 全是边角色未找到内容，按原图返回
            return image;
        }
        Rect rect = Imgproc.boundingRect(nonZero);

        // 裁切，必须clone，否则原始Mat被释放时影响submat
        return matManager.cloneMat(matManager.newMat(image, rect));
    }

    // 统计列表里众数
    private static Scalar mostCommonColor(List<Scalar> scalars) {
        Map<String, Integer> colorCount = new HashMap<>();
        String maxKey = null;
        int maxCount = 0;

        for (Scalar s : scalars) {
            String key = Arrays.toString(s.val);
            int count = colorCount.getOrDefault(key, 0) + 1;
            colorCount.put(key, count);
            if (count > maxCount) {
                maxCount = count;
                maxKey = key;
            }
        }
        // 找到maxKey后转回Scalar
        String[] vals = maxKey.replace("[", "").replace("]", "").split(", ");
        double[] dvals = Arrays.stream(vals).mapToDouble(Double::parseDouble).toArray();
        return new Scalar(dvals);
    }

    /**
     * 对灰度float32图片归一化: (x - mean) / std
     * @param src 灰度图像 (float32, 单通道)
     * @param mean 均值
     * @param std 标准差
     * @return 标准化后的Mat (float32, 单通道)
     */
    public static Mat normalize(MatManager matManager, Mat src, double mean, double std) {
        if (src.channels() != 1 || src.type() != CvType.CV_32F) {
            throw new IllegalArgumentException("输入必须是 float32 单通道灰度图");
        }
        Mat dst = matManager.newMat();
        // 先减去均值
        Core.subtract(src, new Scalar(mean), dst);
        // 再除以std
        Core.divide(dst, new Scalar(std), dst);
        return dst;
    }

    public static Mat toFloat32(MatManager matManager, Mat mat) {
        Mat matFloat32 = matManager.newMat();
        mat.convertTo(matFloat32, CvType.CV_32F);
        return matFloat32;
    }

    /**
     * Resize image following logic:
     * - 短边缩放为 (fixedImgSizeMinus1)
     * - 长边不得超过 maxSize
     * - 双三次插值
     * - 支持 antialias
     *
     * @param src 原始图像
     * @param fixedImgSizeMinus1 缩放后短边（如 255 已是 256-1）
     * @param maxSize            长边最大值
     * @param antialias          是否抗锯齿
     * @return Resize 后的 Mat
     */
    public static Mat resize(MatManager matManager, Mat src, int fixedImgSizeMinus1, int maxSize, boolean antialias) {
        int h = src.rows();
        int w = src.cols();

        // 1. 计算缩放比例
        int minOriginal = Math.min(h, w);
        int maxOriginal = Math.max(h, w);

        float scale = ((float) fixedImgSizeMinus1) / minOriginal;
        // 检查缩放后长边是否超出maxSize，若超出再按maxSize限制缩小scale
        if (Math.round(maxOriginal * scale) > maxSize) {
            scale = ((float) maxSize) / maxOriginal;
        }

        int newW = Math.round(w * scale);
        int newH = Math.round(h * scale);

        // 2. 实际缩放
        Mat dst = matManager.newMat();
        int resizeFlags = Imgproc.INTER_CUBIC;
        // TODO support antialias
        Imgproc.resize(src, dst, new Size(newW, newH), 0, 0, resizeFlags);
        return dst;
    }

    public static Mat rgbToGray(MatManager matManager, Mat mat) {
        Mat gray = matManager.newMat();
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY);
        return gray;
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
        Point[] ordered = new Point[]{tl, tr, br, bl};

        // 8) Delegate to your rotate‐crop routine (keeping float precision, matching Python)
        return getRotateCropImage(matManager, mat, ordered);
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

        return getRotateCropImage(matManager, mat, srcPts);
    }

    /**
     * Crop and rotate the input image based on the given four float points (matching Python PaddleX).
     * This overload preserves float precision from minAreaRect, avoiding rounding errors.
     */
    public static Mat getRotateCropImage(MatManager matManager, Mat mat, Point[] srcPts) {
        if (srcPts.length != 4) {
            throw new IllegalArgumentException("points must be 4");
        }

        // Compute crop width & height via Math.hypot
        double wA = Math.hypot(srcPts[0].x - srcPts[1].x, srcPts[0].y - srcPts[1].y);
        double wB = Math.hypot(srcPts[2].x - srcPts[3].x, srcPts[2].y - srcPts[3].y);
        int cropW = (int) Math.max(wA, wB);

        double hA = Math.hypot(srcPts[0].x - srcPts[3].x, srcPts[0].y - srcPts[3].y);
        double hB = Math.hypot(srcPts[1].x - srcPts[2].x, srcPts[1].y - srcPts[2].y);
        int cropH = (int) Math.max(hA, hB);

        if (cropW <= 0 || cropH <= 0) {
            throw new IllegalArgumentException("Invalid crop dimensions: " + cropW + "x" + cropH);
        }

        // Perspective transform (source points are float, matching Python)
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

        // Rotate if tall (matching Python: np.rot90 when h/w >= 1.5)
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
        matManager.release(srcImg);
        return dstImg;
    }

    public static Mat bgrToRgb(MatManager matManager, Mat bgrImg) {
        Mat rgbImg = matManager.newMat();
        Imgproc.cvtColor(bgrImg, rgbImg, Imgproc.COLOR_BGR2RGB);
        matManager.release(bgrImg);
        return rgbImg;
    }

    public static Mat readToRgb(MatManager matManager, String image) {
        Mat bgrImg = matManager.imread(image, Imgcodecs.IMREAD_COLOR_BGR);
        Mat rgbImg = matManager.newMat();
        Imgproc.cvtColor(bgrImg, rgbImg, Imgproc.COLOR_BGR2RGB);
        matManager.release(bgrImg);
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
        // 读取完成后必须释放临时 Mat，否则会残留在 MatManager 跟踪表中（内存泄露）
        matManager.release(mat);
        return manager.create(ByteBuffer.wrap(buf), shape, DataType.UINT8);
    }

    public static NDArray rgbToNDArray(Mat mat, NDManager manager) {
        byte[] buf = new byte[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(ByteBuffer.wrap(buf), shape, DataType.UINT8);
    }

}
