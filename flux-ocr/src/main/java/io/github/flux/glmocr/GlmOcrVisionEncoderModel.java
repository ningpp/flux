/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.glmocr;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

/**
 * Vision encoder model for GLM-OCR.
 * Takes preprocessed image tensors and outputs image features/embeddings.
 */
public class GlmOcrVisionEncoderModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public GlmOcrVisionEncoderModel(final String modelFile,
                                    final int gpuIndex,
                                    final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Encode images to get vision features.
     *
     * @param pixelValues list of preprocessed image NDArrays [C, H, W]
     * @return image features [batch, num_patches, hidden_size]
     */
    public float[][][] predict(List<NDArray> pixelValues) throws OrtException {
        NDList ndList = new NDList(pixelValues);
        NDArray inputNdArray = NDArrays.stack(ndList);
        long[] shape = inputNdArray.getShape().getShape();
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
             Result result = session.run(Map.of("pixel_values", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
