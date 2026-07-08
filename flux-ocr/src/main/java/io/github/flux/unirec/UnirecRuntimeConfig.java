package io.github.flux.unirec;

import java.util.Map;

record UnirecRuntimeConfig(int encoderGpuIndex, int decoderGpuIndex, int maxTokens) {

    static final int DEFAULT_MAX_TOKENS = 2048;

    static UnirecRuntimeConfig from(int gpuIndex, Map<String, Object> customParams) {
        return new UnirecRuntimeConfig(
                getInt(customParams, gpuIndex, "unirec.encoderGpuIndex", "encoderGpuIndex"),
                getInt(customParams, gpuIndex, "unirec.decoderGpuIndex", "decoderGpuIndex"),
                Math.max(1, getInt(customParams, DEFAULT_MAX_TOKENS, "unirec.maxTokens", "maxTokens"))
        );
    }

    private static int getInt(Map<String, Object> params, int defaultValue, String... keys) {
        if (params == null || params.isEmpty()) {
            return defaultValue;
        }
        for (String key : keys) {
            Object value = params.get(key);
            if (value instanceof Number number) {
                return number.intValue();
            }
            if (value instanceof String stringValue && !stringValue.isBlank()) {
                return Integer.parseInt(stringValue.trim());
            }
        }
        return defaultValue;
    }
}
