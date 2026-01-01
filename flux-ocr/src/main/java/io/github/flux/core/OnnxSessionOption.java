package io.github.flux.core;

import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

public record OnnxSessionOption(int gpuIndex,
                                ExecutionMode executionMode,
                                OptLevel optimizationLevel,
                                int interOpNumThreads,
                                int intraOpNumThreads) {
}
