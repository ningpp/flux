package io.github.flux.core;

public record ObjectDetectionResult(int clsId,
                                    String label,
                                    float score,
                                    float[] coordinate) {
}
