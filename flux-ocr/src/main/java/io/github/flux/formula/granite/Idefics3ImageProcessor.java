// this code is convert from https://github.com/huggingface/transformers
// transformers's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.granite;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import io.github.flux.core.MatManager;
import io.github.flux.formula.pix2text.DeiTImageProcessor;
import io.github.flux.paddle.processor.NougatImageProcessor;
import io.github.flux.paddle.processor.ResizeNdArray;
import io.github.flux.util.IOUtil;

import java.util.ArrayList;
import java.util.List;

public class Idefics3ImageProcessor {

    public static Idefics3PreProcessResult process(HuggingFaceTokenizer tokenizer, String requestText,
                                                   MatManager matManager, NDArray image, NDManager manager) {
        int image_seq_len = 64;
        boolean add_special_tokens = true;
        Idefics3ImageProcessResult imageProcessResult = process_image(matManager, image, manager);
        int[][] image_rows = imageProcessResult.images_list_rows();
        int[][] image_cols = imageProcessResult.images_list_cols();

        String fake_image_token = "<fake_token_around_image>";
        String image_token = "<image>";
        String global_img_token = "<global-img>";
        String sample = requestText;
        int[] sample_rows = image_rows[0];
        int[] sample_cols = image_cols[0];
        // Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
        List<String> image_prompt_strings = new ArrayList<>();
        for (int i = 0; i < sample_rows.length; i++) {
            int n_rows = sample_rows[i];
            int n_cols = sample_cols[i];
            String image_prompt_string = getImagePromptString(n_rows, n_cols,
                    image_seq_len, fake_image_token, image_token, global_img_token);
            image_prompt_strings.add(image_prompt_string);
        }

        String[] split_sample = sample.split(image_token);
        if (split_sample.length == 0) {
            throw new IllegalArgumentException("The image token should be present in the text.");
        }
        // Place in the image prompt strings where the image tokens are
        StringBuilder resultSample = new StringBuilder(split_sample[0]);
        for (int i = 0; i < image_prompt_strings.size(); i++) {
            resultSample.append(image_prompt_strings.get(i)).append(split_sample[i + 1]);
        }

        Encoding encoding = tokenizer.encode(resultSample.toString(), add_special_tokens, false);
        return new Idefics3PreProcessResult(imageProcessResult.pixel_values(),
                imageProcessResult.pixel_attention_mask(),
                encoding.getIds(), encoding.getAttentionMask());
    }

    private static String promptSplitImage(int imageSeqLen,
                                           int imageRows,
                                           int imageCols,
                                           String fakeTokenAroundImage,
                                           String imageToken,
                                           String globalImgToken) {

        StringBuilder textSplitImages = new StringBuilder();

        for (int nH = 0; nH < imageRows; nH++) {
            for (int nW = 0; nW < imageCols; nW++) {
                textSplitImages.append(fakeTokenAroundImage)
                        .append("<row_").append(nH + 1).append("_col_").append(nW + 1).append(">")
                        .append(imageToken.repeat(imageSeqLen));
            }
            textSplitImages.append("\n");
        }

        textSplitImages.append("\n")
                .append(fakeTokenAroundImage)
                .append(globalImgToken)
                .append(imageToken.repeat(imageSeqLen))
                .append(fakeTokenAroundImage);

        return textSplitImages.toString();
    }

    private static String promptSingleImage(int imageSeqLen,
                                            String fakeTokenAroundImage,
                                            String imageToken,
                                            String globalImgToken) {
        return fakeTokenAroundImage
                + globalImgToken
                + imageToken.repeat(imageSeqLen)
                + fakeTokenAroundImage;
    }

    private static String getImagePromptString(int imageRows,
                                               int imageCols,
                                               int imageSeqLen,
                                               String fakeTokenAroundImage,
                                               String imageToken,
                                               String globalImgToken) {

        if (imageRows == 0 && imageCols == 0) {
            return promptSingleImage(imageSeqLen, fakeTokenAroundImage, imageToken, globalImgToken);
        }
        return promptSplitImage(imageSeqLen, imageRows, imageCols,
                fakeTokenAroundImage, imageToken, globalImgToken);
    }

    private static Idefics3ImageProcessResult process_image(MatManager matManager, NDArray image, NDManager manager) {
        int resolution_max_side = 2048;
        float[] imageMean = new float[]{0.5f, 0.5f, 0.5f};
        float[] imageStd = new float[]{0.5f, 0.5f, 0.5f};
        float rescaleFactor = 0.00392156862745098f;
        NDArray resized = resize(matManager, image, resolution_max_side);
        SplitResult splitResult = do_image_splitting(matManager, resized, 512, 1);
        NDList splited = splitResult.frames();
        int[][] images_list_rows = new int[1][];
        int[][] images_list_cols = new int[1][];
        images_list_rows[0] = new int[] {splitResult.num_splits_h};
        images_list_cols[0] = new int[] {splitResult.num_splits_w};

        NDList rgb_list = convert_to_rgb(splited, manager);
        NDList rescaled_list = rescale(rgb_list, rescaleFactor);
        NDList normalized_list = normalize(rescaled_list, imageMean, imageStd, manager);
        PadResult pad_result = pad(matManager, normalized_list, manager);
        NDList images_list = pad_result.padded_images_list();
        NDList padded_mask = pad_result.padded_masks();
        NDList chw_images = toCHW(images_list);
        NDArray pixel_values = NDArrays.stack(chw_images);
        NDArray pixel_attention_mask = NDArrays.stack(padded_mask);

        IOUtil.close(resized);
        IOUtil.close(splited);
        IOUtil.close(rgb_list);
        IOUtil.close(rescaled_list);
        IOUtil.close(normalized_list);
        IOUtil.close(images_list);
        IOUtil.close(padded_mask);
        IOUtil.close(chw_images);
        return new Idefics3ImageProcessResult(
                pixel_values,
                pixel_attention_mask,
                images_list_rows,
                images_list_cols
        );
    }

    // Convert to channels-first format (C, H, W)
    private static NDList toCHW(NDList list) {
        NDList results = new NDList();
        for (NDArray image : list) {
            results.add(image.transpose(2, 0, 1));
        }
        return results;
    }

    private static PadResult pad(MatManager matManager, NDList list, NDManager manager) {
        int[] pad_size = get_max_height_width(list);

        NDList padded_images_list = new NDList();
        NDList padded_masks = new NDList();
        for (NDArray image : list) {
            padded_images_list.add(NougatImageProcessor.pad(matManager, image, pad_size[0], pad_size[1]));
            padded_masks.add(make_pixel_mask(image, pad_size[0], pad_size[1], manager));
        }
        return new PadResult(padded_images_list, padded_masks);
    }

    private static record PadResult(NDList padded_images_list, NDList padded_masks) {
    }

    private static NDArray make_pixel_mask(NDArray image, int outputH, int outputW, NDManager manager) {
        Shape shape = image.getShape();
        long inputH = shape.get(0);
        long inputW = shape.get(1);

        // shape = (outputH, outputW)
        long[][] onesRegion = new long[][]{
                {0, inputH - 1},
                {0, inputW - 1}
        };

        // create zero mask
        NDArray mask = manager.zeros(new Shape(outputH, outputW), DataType.INT64);

        // fill valid region with 1
        NDArray ones = manager.ones(new Shape(inputH, inputW), DataType.INT64);
        mask.set(new NDIndex("0:" + inputH + ",0:" + inputW), ones);

        return mask;
    }

    private static int[] get_max_height_width(NDList list) {
        int maxHeight = -1;
        int maxWidth = -1;
        for (NDArray image : list) {
            long[] shape = image.getShape().getShape();
            int height = (int) shape[0];
            int width = (int) shape[1];
            maxHeight = Math.max(height, maxHeight);
            maxWidth = Math.max(width, maxWidth);
        }
        return new int[]{maxHeight, maxWidth};
    }

    private static NDList normalize(NDList list, float[] mean, float[] std, NDManager manager) {
        NDList results = new NDList();
        for (NDArray image : list) {
            results.add(DeiTImageProcessor.normalize(image, mean, std, manager));
        }
        return results;
    }

    private static NDList rescale(NDList list, float rescaleFactor) {
        NDList results = new NDList();
        for (NDArray image : list) {
            results.add(DeiTImageProcessor.rescale(image, rescaleFactor));
        }
        return results;
    }

    private static SplitResult do_image_splitting(MatManager matManager, NDArray image, int vision_encoder_max_size, int resample) {
        // We first resize both height and width of each image to the nearest max_image_size multiple,
        // disregarding the aspect ratio
        NDArray forVisonResized = resize_for_vision_encoder(matManager, image, vision_encoder_max_size, resample);
        SplitResult splitResult = split_image(matManager, forVisonResized, vision_encoder_max_size);
        return splitResult;
    }

    private static record SplitResult(NDList frames, int num_splits_h, int num_splits_w) {
    }

    private static SplitResult split_image(MatManager matManager, NDArray image, int max_image_size) {
        long[] shape = image.getShape().getShape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        int max_height = max_image_size;
        int max_width = max_image_size;
        NDList frames = new NDList();
        int num_splits_h;
        int num_splits_w;
        if (height > max_height || width > max_width) {
            // Calculate the number of splits
            num_splits_h = Double.valueOf(Math.ceil(((double) height) / ((double) max_height))).intValue();
            num_splits_w = Double.valueOf(Math.ceil(((double) width) / ((double) max_width))).intValue();
            // Calculate the optimal width and height for the sub-images
            int optimal_height = Double.valueOf(Math.ceil(((double) height) / ((double) num_splits_h))).intValue();
            int optimal_width = Double.valueOf(Math.ceil(((double) width) / ((double) num_splits_w))).intValue();
            for (int r = 0; r < num_splits_h; r++) {
                for (int c = 0; c < num_splits_w; c++) {
                    // Calculate the starting point of the crop
                    int start_x = c * optimal_width;
                    int start_y = r * optimal_height;
                    // Calculate the ending point of the crop
                    int end_x = Math.min(start_x + optimal_width, width);
                    int end_y = Math.min(start_y + optimal_height, height);
                    // Crop the image
                    NDArray cropped_image = _crop(
                            image,
                            start_x,
                            start_y,
                            end_x,
                            end_y
                    );
                    frames.add(cropped_image);

                }
            }

            // For the global image at the end, we resize it to match the max_image_size, for cpu memory efficiency
            int global_image_height = max_height;
            int global_image_width = max_width;
            if (height != global_image_height || width != global_image_width) {
                frames.add(new ResizeNdArray(global_image_width, global_image_height, 1).process(matManager, List.of(image)).get(0));
            }
        } else {
            num_splits_h = 0;
            num_splits_w = 0;
            frames.add(image);
        }
        return new SplitResult(frames, num_splits_h, num_splits_w);
    }

    private static NDArray _crop(NDArray image, int w1, int h1, int w2, int h2) {
        // Python image[h1:h2, w1:w2, :]
        NDIndex index = new NDIndex(
                h1 + ":" + h2 + "," +
                        w1 + ":" + w2 + ",:"
        );
        return image.get(index);
    }

    private static NDList convert_to_rgb(NDList list, NDManager manager) {
        NDList results = new NDList();
        for (NDArray image : list) {
            results.add(convert_to_rgb(image, manager));
        }
        return results;
    }

    private static NDArray convert_to_rgb(NDArray image, NDManager manager) {
        Shape shape = image.getShape();
        if (shape.dimension() != 3) {
            throw new IllegalArgumentException("Expect image shape (H, W, C). Got: " + shape);
        }
        long h = shape.get(0);
        long w = shape.get(1);
        long c = shape.get(2);

        if (c == 3) {
            // Already RGB — return as-is (or .clone() if you need a separate array)
            return image;
        }

        if (c != 4) {
            throw new IllegalArgumentException("Image must have 3 (RGB) or 4 (RGBA) channels. Got: " + c);
        }

        // Preserve original dtype to convert back later
        DataType origType = image.getDataType();

        // Work in float to compute alpha blending
        NDArray imgFloat = image.toType(DataType.FLOAT32, true); // copy if necessary

        // Split RGB and alpha
        NDArray rgb = imgFloat.get(new NDIndex(":,:,0:3"));         // (H, W, 3)
        NDArray alpha = imgFloat.get(new NDIndex(":,:,3"));         // (H, W)

        // Normalize alpha to [0,1]
        NDArray alphaNorm = alpha.div(255.0f);                      // (H, W)
        NDArray alphaExpanded = alphaNorm.expandDims(2);            // (H, W, 1)

        // White background (float)
        NDArray white = manager.ones(new Shape(h, w, 3)).mul(255.0f); // (H, W, 3)

        // Composite: out = fg * alpha + bg * (1 - alpha)
        NDArray one = manager.create(1.0f);
        NDArray invAlpha = one.sub(alphaExpanded); // 1 - alpha

        NDArray outFloat = rgb.mul(alphaExpanded).add(white.mul(invAlpha));
        // Convert back to original dtype if necessary (commonly uint8)
        if (origType == DataType.UINT8 || origType == DataType.INT8 || origType == DataType.INT32) {
            // clamp and convert to integer types
            NDArray clipped = outFloat.clip(0f, 255f);
            return clipped.toType(origType, false);
        } else {
            // keep float types
            return outFloat.toType(origType, false);
        }
    }

    private static NDArray resize_for_vision_encoder(MatManager matManager, NDArray image, int vision_encoder_max_size, int resample) {
        long[] shape = image.getShape().getShape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        double aspect_ratio = ((double) width) / ((double) height);
        if (width >= height) {
            width = Double.valueOf(Math.ceil(((double) width) / ((double) vision_encoder_max_size))).intValue()
                    * vision_encoder_max_size;
            height = Double.valueOf(width / aspect_ratio).intValue();
            height = Double.valueOf(Math.ceil(((double) height) / ((double) vision_encoder_max_size))).intValue()
                    * vision_encoder_max_size;
        } else {
            height = Double.valueOf(Math.ceil(((double) height) / ((double) vision_encoder_max_size))).intValue()
                    * vision_encoder_max_size;
            width = Double.valueOf(height * aspect_ratio).intValue();
            width = Double.valueOf(Math.ceil(((double) width) / ((double) vision_encoder_max_size))).intValue()
                    * vision_encoder_max_size;
        }
        return new ResizeNdArray(width, height, resample).process(matManager, List.of(image)).get(0);
    }

    private static NDArray resize(MatManager matManager, NDArray image, int resolution_max_side) {
        int[] heightAndWidth = get_resize_output_image_size(image, resolution_max_side);
        return new ResizeNdArray(heightAndWidth[1], heightAndWidth[0], 1).process(matManager, List.of(image)).get(0);
    }

    private static final int MAX_IMAGE_SIZE = 4096;

    private static int[] get_resize_output_image_size(NDArray image, int resolution_max_side) {
        long[] shape = image.getShape().getShape();
        int height = (int) shape[0];
        int width = (int) shape[1];

        // Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
        int[] hw = _resize_output_size_rescale_to_max_len(height, width, resolution_max_side);
        height = hw[0];
        width = hw[1];

        // Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
        return _resize_output_size_scale_below_upper_bound(height, width, MAX_IMAGE_SIZE);
    }

    private static int[] _resize_output_size_scale_below_upper_bound(int height, int width, Integer max_len) {
        if (max_len == null) {
            max_len = Math.max(height, width);
        }
        double aspect_ratio = ((double) width) / ((double) height);
        if (width >= height && width > max_len) {
            width = max_len;
            height = Double.valueOf(width / aspect_ratio).intValue();
        } else if (height > width && height > max_len) {
            height = max_len;
            width = Double.valueOf(height * aspect_ratio).intValue();
        }
        // Avoid resizing to a size smaller than 1
        return new int[]{
                Math.max(height, 1),
                Math.max(width, 1)
        };
    }

    private static int[] _resize_output_size_rescale_to_max_len(int height, int width, Integer max_len) {
        return _resize_output_size_rescale_to_max_len(height, width, 1, max_len);
    }

    private static int[] _resize_output_size_rescale_to_max_len(int height, int width, int min_len, Integer max_len) {
        if (max_len == null) {
            max_len = Math.max(height, width);
        }
        double aspect_ratio = ((double) width) / ((double) height);
        if (width >= height) {
            width = max_len;
            height = Double.valueOf(width / aspect_ratio).intValue();
            if (height % 2 != 0) {
                height += 1;
            }
        } else {
            height = max_len;
            width = Double.valueOf(height * aspect_ratio).intValue();
            if (width % 2 != 0) {
                width += 1;
            }
        }

        // Avoid resizing to a size smaller than min_len
        return new int[]{
                Math.max(height, min_len),
                Math.max(width, min_len)
        };
    }

}
