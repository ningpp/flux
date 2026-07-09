package io.github.flux.formula.pix2text;

import org.junit.jupiter.api.Test;

import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class Pix2TextDecoderModelTest {

    @Test
    void flattenInputIdsUsesRowMajorBatchSequenceLayout() {
        long[][] inputIds = {
                {10, 11, 12},
                {20, 21, 22}
        };

        long[] flat = Pix2TextDecoderModel.flattenInputIds(inputIds, 2);

        assertArrayEquals(new long[]{10, 11, 20, 21}, flat);
    }

    @Test
    void argmaxLastLogitsReadsOnlyCurrentBatchLastSequenceRow() {
        FloatBuffer logits = FloatBuffer.wrap(new float[]{
                // batch 0, seq 0: should be ignored even though token 0 is large
                99f, 1f, 2f, 3f,
                // batch 0, seq 1: should be ignored
                1f, 98f, 2f, 3f,
                // batch 0, seq 2: token 2 wins
                1f, 2f, 97f, 3f,
                // batch 1, seq 0: should be ignored
                4f, 5f, 6f, 96f,
                // batch 1, seq 1: should be ignored
                95f, 5f, 6f, 7f,
                // batch 1, seq 2: token 1 wins
                4f, 94f, 6f, 7f
        });

        long first = Pix2TextDecoderModel.argmaxLastLogits(logits, 0, 3, 4);
        long second = Pix2TextDecoderModel.argmaxLastLogits(logits, 1, 3, 4);

        assertArrayEquals(new long[]{2L, 1L}, new long[]{first, second});
    }
}
