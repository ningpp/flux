package io.github.flux.core;

import org.opencv.core.Mat;

/**
 *
 * @param ori_img_size     [size_w, size_h]
 * @param result_img       resize后的图像
 * @param img_size         [size_w, size_h]
 * @param scale_factors    [w_scale, h_scale]
 */
public record ObjectDetectionResizeResult(int[] ori_img_size, Mat result_img, int[] img_size, double[] scale_factors) {
}
