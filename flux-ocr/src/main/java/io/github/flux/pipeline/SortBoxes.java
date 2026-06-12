package io.github.flux.pipeline;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public interface SortBoxes {

    NDArray sort(NDManager manager, NDArray dtPolys);

}
