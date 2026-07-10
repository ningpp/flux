package io.github.flux.util;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OnnxSessionUtilTest {

    @AfterEach
    void clearGpuMemLimitProperty() {
        System.clearProperty(OnnxSessionUtil.CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY);
    }

    @Test
    void resolveCudaGpuMemLimitIsUnlimitedByDefault() {
        assertEquals(0L, OnnxSessionUtil.resolveCudaGpuMemLimitBytes());
    }

    @Test
    void resolveCudaGpuMemLimitCanBeOverriddenBySystemProperty() {
        System.setProperty(OnnxSessionUtil.CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY, "536870912");

        assertEquals(536870912L, OnnxSessionUtil.resolveCudaGpuMemLimitBytes());
    }

    @Test
    void resolveCudaGpuMemLimitCanBeDisabledWithZero() {
        System.setProperty(OnnxSessionUtil.CUDA_GPU_MEM_LIMIT_BYTES_PROPERTY, "0");

        assertEquals(0L, OnnxSessionUtil.resolveCudaGpuMemLimitBytes());
    }
}
