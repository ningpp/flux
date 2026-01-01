package io.github.flux.util;

import ai.onnxruntime.OnnxTensor;

import java.util.Map;

public final class OnnxUtil {

    private OnnxUtil() {
    }

    public static void closeTensors(Map<String, OnnxTensor> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        inputs.forEach((k, tensor) -> {
            try {
                if (tensor != null && !tensor.isClosed()) {
                    tensor.close();
                }
            } catch (Exception e) {
                // ignore
            }
        });
    }

}
