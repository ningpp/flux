package io.github.flux.llamajcpp;

import io.github.flux.exception.FluxException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;

class LlamaJCppOcrModelTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldRejectUnsupportedModelNameBeforeRuntimeInitialization() {
        assertThrows(FluxException.class,
                () -> LlamaJCppOcrModel.getSharedInstance(tempDir.toString(), "UnsupportedModel", -1, null, Map.of()));
    }

    @Test
    void shouldFailFastWhenModelFilesAreMissing() throws Exception {
        Files.createDirectories(tempDir.resolve("LlamaJCpp-OCR"));

        assertThrows(FluxException.class,
                () -> LlamaJCppOcrModel.getSharedInstance(tempDir.toString(), "LlamaJCpp-OCR", -1, null, Map.of()));
    }
}
