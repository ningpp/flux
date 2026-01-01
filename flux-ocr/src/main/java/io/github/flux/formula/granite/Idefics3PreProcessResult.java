package io.github.flux.formula.granite;

import ai.djl.ndarray.NDArray;

public record Idefics3PreProcessResult(NDArray pixel_values,
                                       NDArray pixel_attention_mask,
                                       long[] input_ids,
                                       long[] attention_mask) {
}
