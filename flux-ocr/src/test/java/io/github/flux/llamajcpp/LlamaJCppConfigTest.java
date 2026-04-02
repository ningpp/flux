package io.github.flux.llamajcpp;

import io.github.flux.exception.FluxException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LlamaJCppConfigTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldResolveDefaultsFromModelDirectory() throws Exception {
        Path modelDir = Files.createDirectories(tempDir.resolve("LlamaJCpp-OCR"));
        Path modelFile = Files.createFile(modelDir.resolve("model.gguf"));
        Path mmprojFile = Files.createFile(modelDir.resolve("mmproj.gguf"));

        LlamaJCppConfig config = LlamaJCppConfig.from(tempDir.toString(), "LlamaJCpp-OCR", -1, Map.of());

        assertEquals(modelFile.toAbsolutePath(), config.modelFile());
        assertEquals(mmprojFile.toAbsolutePath(), config.mmprojFile());
        assertEquals(8192, config.contextSize());
        assertEquals(Math.min(8192, 2048), config.batchSize());
        assertFalse(config.useGpu());
        assertTrue(config.useChatTemplate());
        assertEquals("<IMG>\nRead all text in the image.", config.buildPromptContent(null));
    }

    @Test
    void shouldApplyCustomParameters() throws Exception {
        Path modelDir = Files.createDirectories(tempDir.resolve("LlamaJCpp-OCR"));
        Path customModel = Files.createFile(modelDir.resolve("custom-model.gguf"));
        Path customMmproj = Files.createFile(modelDir.resolve("projector.gguf"));

        LlamaJCppConfig config = LlamaJCppConfig.from(tempDir.toString(), "LlamaJCpp-OCR", 0, Map.of(
                "modelFile", "custom-model.gguf",
                "mmprojFile", "projector.gguf",
                "contextSize", 4096,
                "batchSize", 1024,
                "maxTokens", 256,
                "nGpuLayers", 32,
                "nThreads", 6,
                "temperature", 0.7f,
                "topK", 20,
                "topP", 0.8f,
                "minP", 0.1f,
                "seed", 7,
                "useGpu", true,
                "useNativeImageDecoder", false,
                "printTimings", true,
                "useChatTemplate", false,
                "mediaMarker", "<MEDIA>",
                "systemPrompt", "ocr system",
                "promptTemplate", "{media}\nPrompt: {prompt}",
                "stopStrings", List.of("</s>", "<END>")
        ));

        assertEquals(customModel.toAbsolutePath(), config.modelFile());
        assertEquals(customMmproj.toAbsolutePath(), config.mmprojFile());
        assertEquals(4096, config.contextSize());
        assertEquals(1024, config.batchSize());
        assertEquals(256, config.maxTokens());
        assertEquals(32, config.nGpuLayers());
        assertEquals(6, config.nThreads());
        assertEquals(0.7f, config.temperature());
        assertEquals(20, config.topK());
        assertEquals(0.8f, config.topP());
        assertEquals(0.1f, config.minP());
        assertEquals(7, config.seed());
        assertTrue(config.useGpu());
        assertFalse(config.useNativeImageDecoder());
        assertTrue(config.printTimings());
        assertFalse(config.useChatTemplate());
        assertEquals("ocr system", config.systemPrompt());
        assertEquals(List.of("</s>", "<END>"), config.stopStrings());
        assertEquals("<MEDIA>\nPrompt: OCR now", config.buildPromptContent("OCR now"));
    }

    @Test
    void shouldFailWhenRequiredFilesAreMissing() throws Exception {
        Files.createDirectories(tempDir.resolve("LlamaJCpp-OCR"));

        FluxException error = assertThrows(FluxException.class,
                () -> LlamaJCppConfig.from(tempDir.toString(), "LlamaJCpp-OCR", -1, Map.of()));

        assertTrue(error.getMessage().contains("GGUF model file"));
    }
}
