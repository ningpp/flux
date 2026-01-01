package io.github.flux.core;

import java.util.List;

public record FormulaRecognitionResult(List<String> formulas, long[] tokens, float score) {
}
