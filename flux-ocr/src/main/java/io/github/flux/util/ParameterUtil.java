package io.github.flux.util;

import io.github.flux.paddle.processor.LimitType;

import java.util.Map;

public final class ParameterUtil {

    private ParameterUtil() {
    }

    public static Float getFloat(Map<String, Object> params, String key) {
        Number num = getNumber(params, key);
        return num == null ? null : num.floatValue();
    }

    public static Integer getInteger(Map<String, Object> params, String key) {
        Number num = getNumber(params, key);
        return num == null ? null : num.intValue();
    }

    public static Number getNumber(Map<String, Object> params, String key) {
        Object obj = params == null ? null : params.get(key);
        return obj == null ? null : (obj instanceof Number num ? num : Double.valueOf(obj.toString()));
    }

    public static LimitType getLimitType(Map<String, Object> params, String key) {
        Object obj = params == null ? null : params.get(key);
        if (obj instanceof String str) {
            for (var type : LimitType.values()) {
                if ( type.name().equalsIgnoreCase(str)) {
                    return type;
                }
            }
        }
        return null;
    }
}
