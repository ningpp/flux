package io.github.flux.core;

import ai.onnxruntime.OrtEnvironment;

public record ModelParam(String modelRootDir,
                         String modelName,
                         int gpuIndex,
                         OrtEnvironment env) {
}
