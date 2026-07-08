package io.github.flux.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrackedOnnxResourceLifecycleTest {

    private static final Pattern TRACKED_RESOURCE_IN_TRY_WITH = Pattern.compile(
            "try\\s*\\([^)]*matManager\\.(?:createOnnxTensor|runSession)",
            Pattern.DOTALL);

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
