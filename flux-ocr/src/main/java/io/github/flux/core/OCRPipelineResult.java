package io.github.flux.core;

import java.util.List;

public record OCRPipelineResult(int[][] detPolys, List<RecognitionResult> recResults) {
}
