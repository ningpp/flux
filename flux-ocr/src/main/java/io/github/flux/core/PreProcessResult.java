package io.github.flux.core;

import ai.djl.ndarray.NDArray;
import org.opencv.core.Mat;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

public record PreProcessResult(Mat mat, NDArray ndArray) {

    public static List<NDArray> getNDArrays(Collection<PreProcessResult> pprs) {
        if (pprs == null) {
            return List.of();
        }
        return pprs.stream().map(PreProcessResult::ndArray).filter(Objects::nonNull).toList();
    }

    public static List<Mat> getMats(Collection<PreProcessResult> pprs) {
        if (pprs == null) {
            return List.of();
        }
        return pprs.stream().map(PreProcessResult::mat).filter(Objects::nonNull).toList();
    }

}
