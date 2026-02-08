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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.Map;

/**
 * Embedding model for GLM-OCR.
 * Converts token IDs to embeddings.
 */
public class GlmOcrEmbedModel implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public GlmOcrEmbedModel(final String modelFile,
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
     * Convert token IDs to embeddings.
     *
     * @param inputIds token IDs [batch, seq_len]
     * @return embeddings [batch, seq_len, hidden_size]
     */
    public OnnxTensor predictTensor(long[][] inputIds) throws OrtException {
        try (OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env)) {
            Result result = session.run(Map.of("input_ids", onnxInput));
            return (OnnxTensor) result.get(0);
        }
    }

    /**
     * Convert token IDs to embeddings.
     *
     * @param inputIds token IDs [batch, seq_len]
     * @return embeddings [batch, seq_len, hidden_size]
     */
    public float[][][] predict(long[][] inputIds) throws OrtException {
        try (OnnxTensor onnxInput = ArrayUtil.createOnnxTensor(inputIds, env);
             Result result = session.run(Map.of("input_ids", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
