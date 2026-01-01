package io.github.flux.core;

import java.util.List;

public record TextDetectionResult(int[][][] polys, List<Float> scores) {
}
