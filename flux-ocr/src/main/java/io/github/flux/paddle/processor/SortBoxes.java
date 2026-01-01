package io.github.flux.paddle.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public interface SortBoxes {

    NDArray sort(NDManager manager, NDArray dtPolys);

}
