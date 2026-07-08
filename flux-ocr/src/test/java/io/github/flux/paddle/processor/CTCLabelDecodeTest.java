package io.github.flux.paddle.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.RecognitionResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.FloatBuffer;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CTCLabelDecodeTest {

    @TempDir
    Path tempDir;

    @Test
    void rawLogitDecodeMatchesNdArrayDecode() throws Exception {
        Path config = tempDir.resolve("inference.yml");
        Files.writeString(config, """
                PostProcess:
                  character_dict:
                    - A
                    - B
                    - C
                    - D
                """);

        float[][] logits = {
                {0.01f, 0.90f, 0.05f, 0.03f, 0.01f},
                {0.02f, 0.91f, 0.04f, 0.02f, 0.01f},
                {0.95f, 0.01f, 0.02f, 0.01f, 0.01f},
                {0.01f, 0.05f, 0.80f, 0.10f, 0.04f},
                {0.01f, 0.04f, 0.81f, 0.10f, 0.04f},
                {0.02f, 0.03f, 0.04f, 0.85f, 0.06f}
        };

        CTCLabelDecode decoder = new CTCLabelDecode(config.toString(), false, false);
        List<RecognitionResult> fastResults = decoder.process(logits);

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray preds = manager.create(logits);
            List<RecognitionResult> ndResults = decoder.process(preds);

            assertEquals(ndResults.size(), fastResults.size());
            assertEquals(ndResults.getFirst().text(), fastResults.getFirst().text());
            assertArrayEquals(ndResults.getFirst().scores(), fastResults.getFirst().scores(), 1e-7);
        }
    }

    @Test
    void floatBufferDecodeMatchesRawLogitDecode() throws Exception {
        Path config = tempDir.resolve("inference.yml");
        Files.writeString(config, """
                PostProcess:
                  character_dict:
                    - A
                    - B
                    - C
                    - D
                """);

        float[][] logits = {
                {0.01f, 0.90f, 0.05f, 0.03f, 0.01f},
                {0.02f, 0.91f, 0.04f, 0.02f, 0.01f},
                {0.95f, 0.01f, 0.02f, 0.01f, 0.01f},
                {0.01f, 0.05f, 0.80f, 0.10f, 0.04f},
                {0.01f, 0.04f, 0.81f, 0.10f, 0.04f},
                {0.02f, 0.03f, 0.04f, 0.85f, 0.06f}
        };

        int offset = 3;
        int classCount = logits[0].length;
        FloatBuffer buffer = FloatBuffer.allocate(offset + logits.length * classCount);
        buffer.position(offset);
        for (float[] timestep : logits) {
            buffer.put(timestep);
        }

        CTCLabelDecode decoder = new CTCLabelDecode(config.toString(), false, false);
        List<RecognitionResult> rawResults = decoder.process(logits);
        List<RecognitionResult> bufferResults = decoder.process(buffer, offset, logits.length, classCount);

        assertEquals(rawResults.getFirst().text(), bufferResults.getFirst().text());
        assertArrayEquals(rawResults.getFirst().scores(), bufferResults.getFirst().scores(), 1e-7);
    }
}
