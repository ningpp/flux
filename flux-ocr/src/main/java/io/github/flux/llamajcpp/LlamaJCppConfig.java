package io.github.flux.llamajcpp;

import io.github.flux.exception.FluxException;
import io.github.flux.util.ParameterUtil;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Map;

public record LlamaJCppConfig(Path modelFile,
                              Path mmprojFile,
                              int contextSize,
                              int batchSize,
                              int maxTokens,
                              int nGpuLayers,
                              int nThreads,
                              float temperature,
                              int topK,
                              float topP,
                              float minP,
                              int seed,
                              boolean useGpu,
                              boolean useNativeImageDecoder,
                              boolean printTimings,
                              boolean useChatTemplate,
                              String mediaMarker,
                              String systemPrompt,
                              String promptTemplate,
                              List<String> stopStrings) {

    private static final String DEFAULT_MODEL_FILE = "model.gguf";
    private static final String DEFAULT_MMPROJ_FILE = "mmproj.gguf";
    private static final String DEFAULT_SYSTEM_PROMPT = "Extract all readable text from the provided image.";
    private static final String DEFAULT_PROMPT_TEMPLATE = "{{media}}\n{{prompt}}";

    public static LlamaJCppConfig from(final String modelRootDir,
                                       final String modelName,
                                       final int gpuIndex,
                                       final Map<String, Object> customParams) {
        final Map<String, Object> params = customParams == null ? Map.of() : customParams;
        final Path modelDir = Path.of(modelRootDir, modelName);
        final int contextSize = requirePositive("contextSize", getInteger(params, "contextSize", 8192));
        final int batchSize = requirePositive("batchSize", Math.min(getInteger(params, "batchSize", Math.min(contextSize, 2048)), contextSize));
        final int maxTokens = requirePositive("maxTokens", getInteger(params, "maxTokens", 2048));
        final int nThreads = requirePositive("nThreads", getInteger(params, "nThreads", Runtime.getRuntime().availableProcessors()));
        final boolean useGpu = getBoolean(params, gpuIndex >= 0, "useGpu");

        return new LlamaJCppConfig(
                requireFile(modelDir, params, "GGUF model file", DEFAULT_MODEL_FILE, "modelFile", "ggufFile", "modelPath"),
                requireFile(modelDir, params, "mmproj file", DEFAULT_MMPROJ_FILE, "mmprojFile", "projectorFile", "mmprojPath"),
                contextSize,
                batchSize,
                maxTokens,
                Math.max(0, getInteger(params, "nGpuLayers", useGpu ? 999 : 0)),
                nThreads,
                getFloat(params, "temperature", 0.2f),
                getInteger(params, "topK", 40),
                getFloat(params, "topP", 0.95f),
                getFloat(params, "minP", 0.05f),
                getInteger(params, "seed", 42),
                useGpu,
                getBoolean(params, true, "useNativeImageDecoder"),
                getBoolean(params, false, "printTimings"),
                getBoolean(params, true, "useChatTemplate"),
                getString(params, "<IMG>", "mediaMarker"),
                getString(params, DEFAULT_SYSTEM_PROMPT, "systemPrompt"),
                getString(params, DEFAULT_PROMPT_TEMPLATE, "promptTemplate"),
                List.copyOf(getStrings(params, "stopStrings", "stopString"))
        );
    }

    public String buildPromptContent(final String prompt) {
        final String effectivePrompt = prompt == null || prompt.isBlank()
                ? "Read all text in the image."
                : prompt;
        return replacePromptPlaceholders(promptTemplate, effectivePrompt);
    }

    private String replacePromptPlaceholders(final String template, final String prompt) {
        return template
                .replace("{{media}}", mediaMarker)
                .replace("{media}", mediaMarker)
                .replace("{{prompt}}", prompt)
                .replace("{prompt}", prompt);
    }

    private static Path requireFile(final Path modelDir,
                                    final Map<String, Object> params,
                                    final String description,
                                    final String defaultFileName,
                                    final String... keys) {
        final String configuredPath = getNullableString(params, keys);
        final Path file = configuredPath == null || configuredPath.isBlank()
                ? modelDir.resolve(defaultFileName)
                : resolvePath(modelDir, configuredPath);
        if (!Files.isRegularFile(file)) {
            throw new FluxException(description + " not found: " + file.toAbsolutePath());
        }
        return file.toAbsolutePath();
    }

    private static Path resolvePath(final Path modelDir, final String filePath) {
        final Path path = Path.of(filePath);
        return path.isAbsolute() ? path : modelDir.resolve(path);
    }

    private static Integer getInteger(final Map<String, Object> params, final String key, final int defaultValue) {
        final Integer value = ParameterUtil.getInteger(params, key);
        return value == null ? defaultValue : value;
    }

    private static int requirePositive(final String key, final int value) {
        if (value <= 0) {
            throw new FluxException(key + " must be greater than 0");
        }
        return value;
    }

    private static Float getFloat(final Map<String, Object> params, final String key, final float defaultValue) {
        final Float value = ParameterUtil.getFloat(params, key);
        return value == null ? defaultValue : value;
    }

    private static boolean getBoolean(final Map<String, Object> params, final boolean defaultValue, final String... keys) {
        final String value = getNullableString(params, keys);
        return value == null ? defaultValue : Boolean.parseBoolean(value);
    }

    private static String getString(final Map<String, Object> params, final String defaultValue, final String... keys) {
        final String value = getNullableString(params, keys);
        return value == null || value.isBlank() ? defaultValue : value;
    }

    private static String getNullableString(final Map<String, Object> params, final String... keys) {
        if (params == null) {
            return null;
        }
        for (String key : keys) {
            final Object value = params.get(key);
            if (value != null) {
                return String.valueOf(value);
            }
        }
        return null;
    }

    private static List<String> getStrings(final Map<String, Object> params, final String... keys) {
        if (params == null) {
            return List.of();
        }
        for (String key : keys) {
            final Object value = params.get(key);
            if (value instanceof String str) {
                return str.isBlank() ? List.of() : List.of(str);
            }
            if (value instanceof Collection<?> collection) {
                return collection.stream()
                        .map(String::valueOf)
                        .filter(str -> !str.isBlank())
                        .toList();
            }
        }
        return List.of();
    }
}
