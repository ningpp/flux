package io.github.flux.dolphin;

import io.github.flux.paddle.processor.Rescale;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class DolphinPreProcessor {

    public Mat process(Mat rgbImg) {
        int size = 896;
        int input_width = rgbImg.cols();
        int input_height = rgbImg.rows();
        int output_width = size;
        int output_height = size;
        int shortest_edge = size;
        int short_val;
        int long_val;
        if (input_width <= input_height) {
            short_val = input_width;
            long_val = input_height;
        } else {
            short_val = input_height;
            long_val = input_width;
        }
        int requested_new_short = shortest_edge;

        int new_short = requested_new_short;
        int new_long = requested_new_short * long_val / short_val;

        // (new_long, new_short) if width <= height else (new_short, new_long)
        int[] imgSize = new int[2];
        if (input_width <= input_height) {
            imgSize[0] = new_long;
            imgSize[1] = new_short;
        } else {
            imgSize[0] = new_short;
            imgSize[1] = new_long;
        }

        Mat resizedImg = new Resize(imgSize[0], imgSize[1], 2).process(rgbImg);
        rgbImg.release();
        Mat thumbnailImg = thumbnail(resizedImg, size, size);
        resizedImg.release();

        Mat paddedImage = pad(thumbnailImg, size, size);
        thumbnailImg.release();
        double rescale_factor = 0.00392156862745098;
        Mat rescaled = new Rescale(rescale_factor).process(paddedImage);
        paddedImage.release();

        Mat normalized = ImageUtil.normalize(rescaled,
                new Scalar(0.485, 0.456, 0.406),
                new Scalar(0.229, 0.224, 0.225));
        rescaled.release();

        Mat chw = new ToCHWImage().process(normalized);

        normalized.release();
        return chw;
    }

    private Mat pad(Mat img, int output_height, int output_width) {
        int input_width = img.cols();
        int input_height = img.rows();
        int delta_width = output_width - input_width;
        int delta_height = output_height - input_height;
        // Python: delta_height // 2
        int pad_top = Math.floorDiv(delta_height, 2);
        // Python: delta_width // 2
        int pad_left = Math.floorDiv(delta_width, 2);
        int pad_bottom = delta_height - pad_top;
        int pad_right = delta_width - pad_left;
        //padding = ((pad_top, pad_bottom), (pad_left, pad_right))

        Scalar paddingColor = new Scalar(255, 255, 255);
        Mat paddedImage = new Mat();
        Core.copyMakeBorder(
                img,
                paddedImage,
                pad_top,        // top
                pad_bottom,     // bottom
                pad_left,       // left
                pad_right,      // right
                Core.BORDER_CONSTANT,
                paddingColor
        );
        img.release();
        return paddedImage;
    }

    private Mat thumbnail(Mat img, int output_height, int output_width) {
        int input_width = img.cols();
        int input_height = img.rows();
        // We always resize to the smallest of either the input or output size.
        int height = Math.min(input_height, output_height);
        int width = Math.min(input_width, output_width);

        if (height == input_height && width == input_width) {
            return img;
        }
        if (input_height > input_width) {
            width = input_width * height / input_height;
        } else if (input_width > input_height) {
            height = input_height * width / input_width;
        }

        // TODO reducing_gap=2.0
        Mat resizedImg = new Resize(height, width, 2).process(img);
        img.release();
        return resizedImg;
    }

}
