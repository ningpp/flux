package io.github.flux.formula.granite;

import ai.djl.ndarray.NDArray;

public record Idefics3ImageProcessResult(NDArray pixel_values,
                                         NDArray pixel_attention_mask,
                                         int[][] images_list_rows,
                                         int[][] images_list_cols) {
}
