package io.github.flux.unirec;

import org.junit.jupiter.api.Test;

import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;

class UnirecDecoderModelTest {

    @Test
    void argmaxLastLogitsReadsOnlyCurrentBatchLastSequenceRow() {
        FloatBuffer logits = FloatBuffer.wrap(new float[]{
                // batch 0, seq 0: ignored
                90f, 1f, 2f,
                // batch 0, seq 1: ignored
                3f, 89f, 4f,
                // batch 0, seq 2: token 2 wins
                5f, 6f, 88f
        });

        assertEquals(2, UnirecDecoderModel.argmaxLastLogits(logits, 0, 3, 3));
    }
}
