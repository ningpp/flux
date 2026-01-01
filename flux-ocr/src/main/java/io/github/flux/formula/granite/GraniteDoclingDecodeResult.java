package io.github.flux.formula.granite;

import java.util.Map;

public record GraniteDoclingDecodeResult(float[][][] logits, Map<String, float[][][][]> present_key_values) {
}
