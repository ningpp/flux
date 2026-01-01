package io.github.flux.paddle.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import io.github.flux.formula.pix2text.DeiTImageProcessor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;

public class NougatImageProcessor {

    public NDArray process(Mat rgbImg, NDManager manager) {
        NDArray imgNdArray = ImageUtil.toNDArrayUint8(rgbImg, manager);
        NDArray input = process(imgNdArray, manager);

        rgbImg.release();
        imgNdArray.close();
        return input;
    }

    private static int[] get_resize_output_image_size(NDArray image, int shortest_edge) {
        long[] shape = image.getShape().getShape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        int short_v;
        int long_v;
        if (width <= height) {
            short_v = width;
            long_v = height;
        } else {
            short_v = height;
            long_v = width;
        }
        int new_short = shortest_edge;
        int new_long = shortest_edge * long_v / short_v;

        if (width <= height) {
            return new int[]{new_long, new_short};
        } else {
            return new int[]{new_short, new_long};
        }
    }

    private static NDArray process(NDArray image, NDManager manager) {
        float rescaleFactor = 0.00392156862745098f;
        float[] imageMean = new float[]{0.485f, 0.456f, 0.406f};
        float[] imageStd = new float[]{0.229f, 0.224f, 0.225f};
        int height = 224;
        int width = 560;
        NDArray resized = resize(image, Math.min(height, width));
        NDArray thumbnailed = thumbnail(resized, height, width, 2);
        NDArray padded = pad(thumbnailed, height, width);
        NDArray rescaled = DeiTImageProcessor.rescale(padded, rescaleFactor);
        NDArray normalized = DeiTImageProcessor.normalize(rescaled, imageMean, imageStd, manager);

        // Convert to channels-first format (C, H, W)
        NDArray result = normalized.transpose(2, 0, 1);
        normalized.close();
        rescaled.close();
        padded.close();
        thumbnailed.close();
        resized.close();

        return result;
    }

    public static NDArray pad(NDArray image, int output_height, int output_width) {
        long[] shape = image.getShape().getShape();
        int input_height = (int) shape[0];
        int input_width = (int) shape[1];
        int delta_width = output_width - input_width;
        int delta_height = output_height - input_height;
        // Python: delta_height // 2
        int pad_top = Math.floorDiv(delta_height, 2);
        // Python: delta_width // 2
        int pad_left = Math.floorDiv(delta_width, 2);

        int pad_bottom = delta_height - pad_top;
        int pad_right = delta_width - pad_left;
        return image.pad(new Shape(pad_left, pad_right, 0, 0, pad_top, pad_bottom), 0.0);
    }

    private static NDArray resize(NDArray image, int shortest_edge) {
        int[] heightAndWidth = get_resize_output_image_size(image, shortest_edge);
        return resize(image, heightAndWidth[0], heightAndWidth[1], 2);
    }

    private static NDArray resize(NDArray image, int targetHeight, int targetWidth, int resampleMode) {
        return new ResizeNdArray(targetWidth, targetHeight, resampleMode).process(List.of(image)).get(0);
    }

    private static NDArray thumbnail(NDArray image, int output_height, int output_width, int resampleMode) {
        long[] shape = image.getShape().getShape();
        int input_height = (int) shape[0];
        int input_width = (int) shape[1];
        int height = Math.min(input_height, output_height);
        int width = Math.min(input_width, output_width);
        if (height == input_height && width == input_width) {
            return image;
        }

        if (input_height > input_width) {
            width = input_width * height / input_height;
        } else if (input_width > input_height) {
            height = input_height * width / input_width;
        }
        return new ResizeNdArray(width, height, resampleMode).process(List.of(image)).get(0);
    }

}
