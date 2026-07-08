package io.github.flux.unirec;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class UnirecRuntimeConfigTest {

    @Test
    void defaultsUseRequestedDeviceAndLegacyTokenLimit() {
        UnirecRuntimeConfig config = UnirecRuntimeConfig.from(0, Map.of());

        assertEquals(0, config.encoderGpuIndex());
        assertEquals(0, config.decoderGpuIndex());
        assertEquals(2048, config.maxTokens());
    }

    @Test
    void customParamsCanMoveDecoderAndCapGeneration() {
        UnirecRuntimeConfig config = UnirecRuntimeConfig.from(0, Map.of(
                "unirec.decoderGpuIndex", -1,
                "unirec.maxTokens", 768
        ));

        assertEquals(0, config.encoderGpuIndex());
        assertEquals(-1, config.decoderGpuIndex());
        assertEquals(768, config.maxTokens());
    }

    @Test
    void shortKeysAreAcceptedForDirectModelConstruction() {
        UnirecRuntimeConfig config = UnirecRuntimeConfig.from(-1, Map.of(
                "encoderGpuIndex", 0,
                "decoderGpuIndex", -1,
                "maxTokens", "512"
        ));

        assertEquals(0, config.encoderGpuIndex());
        assertEquals(-1, config.decoderGpuIndex());
        assertEquals(512, config.maxTokens());
    }
}
