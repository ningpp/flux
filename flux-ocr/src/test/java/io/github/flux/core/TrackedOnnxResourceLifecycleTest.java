package io.github.flux.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrackedOnnxResourceLifecycleTest {

    private static final Pattern TRACKED_RESOURCE_IN_TRY_WITH = Pattern.compile(
            "try\\s*\\([^)]*matManager\\.(?:createOnnxTensor|runSession)",
            Pattern.DOTALL);
    private static final List<String> UNMANAGED_SESSION_PATTERNS = List.of(
            "new OrtSession.SessionOptions",
            ".addCUDA(",
            "env.createSession("
    );

    @ParameterizedTest
    @MethodSource("ocrPipelineGpuSources")
    void ocrPipelineGpuSourcesReleaseTrackedOrtResourcesThroughMatManager(String sourcePath) throws IOException {
        String source = Files.readString(Path.of("src/main/java").resolve(sourcePath));

        assertFalse(TRACKED_RESOURCE_IN_TRY_WITH.matcher(source).find(),
                sourcePath + " must not put MatManager-tracked ORT resources in try-with-resources; "
                        + "try-with closes the native handle but leaves the Java wrapper tracked until page end");
        assertFalse(source.contains("IOUtil.close(onnxInput)"),
                sourcePath + " must release tracked ONNX inputs with matManager.release(onnxInput)");
        assertFalse(source.contains("IOUtil.close(onnxResult)"),
                sourcePath + " must release tracked ONNX results with matManager.release(onnxResult)");
        assertFalse(source.contains("IOUtil.close(result)"),
                sourcePath + " must release tracked ONNX results with matManager.release(result)");
    }

    @ParameterizedTest
    @MethodSource("ocrPipelineSessionSources")
    void ocrPipelineGpuSourcesCreateSessionsThroughOnnxSessionUtil(String sourcePath) throws IOException {
        String source = Files.readString(Path.of("src/main/java").resolve(sourcePath));

        assertTrue(source.contains("OnnxSessionUtil.createSession"),
                sourcePath + " must use OnnxSessionUtil so CUDA sessions share memory-pattern and arena limits");
        assertFalse(source.contains("new OrtSession.SessionOptions"),
                sourcePath + " must not create unmanaged SessionOptions directly");
        assertFalse(source.contains(".addCUDA("),
                sourcePath + " must not add CUDA directly because that bypasses OrtCUDAProviderOptions limits");
        assertFalse(source.contains("env.createSession("),
                sourcePath + " must not create unmanaged OrtSession directly");
    }

    @Test
    void productionOnnxSourcesCreateSessionsThroughOnnxSessionUtil() throws IOException {
        Path mainJava = Path.of("src/main/java");

        try (Stream<Path> paths = Files.walk(mainJava)) {
            List<String> offenders = paths
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !relativeSourcePath(mainJava, path)
                            .equals("io/github/flux/util/OnnxSessionUtil.java"))
                    .filter(path -> sourceContainsAny(path, UNMANAGED_SESSION_PATTERNS))
                    .map(path -> relativeSourcePath(mainJava, path))
                    .sorted()
                    .toList();

            assertTrue(offenders.isEmpty(),
                    "Production ONNX sources must create sessions through OnnxSessionUtil: " + offenders);
        }
    }

    @Test
    void matManagerReleaseRemovesTrackedCloseableImmediately() throws Exception {
        CountingCloseable closeable = new CountingCloseable();

        try (MatManager matManager = new MatManager()) {
            matManager.track(closeable);

            assertEquals(1, matManager.trackedCloseableCount());
            matManager.release(closeable);

            assertTrue(closeable.closed);
            assertEquals(1, closeable.closeCount);
            assertEquals(0, matManager.trackedCloseableCount());
        }

        assertEquals(1, closeable.closeCount);
    }

    private static Stream<String> ocrPipelineGpuSources() {
        return Stream.of(
                "io/github/flux/formula/pix2text/Pix2TextEncoderModel.java",
                "io/github/flux/formula/pix2text/Pix2TextDecoderModel.java",
                "io/github/flux/paddle/PPDocLayoutV3Model.java",
                "io/github/flux/unirec/UnirecEncoderModel.java",
                "io/github/flux/unirec/UnirecEncoderModelPredictResult.java"
        );
    }

    private static Stream<String> ocrPipelineSessionSources() {
        return Stream.of(
                "io/github/flux/docling/DoclingLayoutModel.java",
                "io/github/flux/paddle/PPDocLayoutV3Model.java",
                "io/github/flux/paddle/predictor/PaddleObjectDetectionPredictor.java"
        );
    }

    private static boolean sourceContainsAny(Path path, List<String> patterns) {
        try {
            String source = Files.readString(path);
            return patterns.stream().anyMatch(source::contains);
        } catch (IOException e) {
            throw new AssertionError("Could not read " + path, e);
        }
    }

    private static String relativeSourcePath(Path root, Path path) {
        return root.relativize(path).toString().replace('\\', '/');
    }

    private static final class CountingCloseable implements AutoCloseable {
        private boolean closed;
        private int closeCount;

        @Override
        public void close() {
            closed = true;
            closeCount++;
        }
    }
}
