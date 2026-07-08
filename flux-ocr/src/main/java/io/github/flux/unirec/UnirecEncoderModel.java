// this code is convert from https://github.com/Topdu/OpenOCR
// OpenOCR's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.util.OnnxSessionUtil;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.Map;

public class UnirecEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public UnirecEncoderModel(final String modelFile,
                              final int gpuIndex,
                              final OrtEnvironment env) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public UnirecEncoderModelPredictResult predict(PreProcessResult ppr, MatManager matManager, NDManager ndManager) {
        OnnxTensor onnxInput = null;
        OrtSession.Result result = null;
        try {
            // The preprocessor already produced a CHW-planar float Mat (see UnirecProcessor /
            // ToCHWImage), i.e. the exact [C, H, W] layout the ONNX model expects. Feed it
            // directly to OnnxTensor to avoid the DJL NDArray -> ByteBuffer -> OnnxTensor copies.
            Mat mat = ppr.mat();
            long[] shape = new long[]{1, mat.channels(), mat.height(), mat.width()};
            int total = mat.height() * mat.width() * mat.channels();
            float[] buf = new float[total];
            mat.get(0, 0, buf);
            // Release the input Mat AND drop it from the MatManager tracking table.
            matManager.release(mat);
            onnxInput = OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), shape);
            result = session.run(Map.of("pixel_values", onnxInput));
            IOUtil.close(onnxInput);
            onnxInput = null;
            return new UnirecEncoderModelPredictResult(
                    result,
                    (OnnxTensor) result.get(0),
                    (OnnxTensor) result.get(1),
                    (OnnxTensor) result.get(2));
        } catch (Exception e) {
            IOUtil.close(result);
            IOUtil.close(onnxInput);
            throw new FluxException(e);
        }
    }

}
