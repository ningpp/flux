package io.github.flux.core;

import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

public record ProcessedMat(int oriWidth, int oriHeight, Mat processed) {

    public void release() {
        IOUtil.close(processed);
    }

}
