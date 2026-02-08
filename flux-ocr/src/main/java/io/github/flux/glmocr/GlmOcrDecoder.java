package io.github.flux.glmocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtException;

public interface GlmOcrDecoder extends AutoCloseable {

    long[][] predict(float[][][] imageFeatures,
                     long[][] inputIds,
                     float[][][] inputsEmbeds,
                     int[] imageGridThw,
                     GlmOcrEmbedModel embedModel,
                     NDManager ndManager) throws OrtException;

}
