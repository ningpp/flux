// this code is convert from https://github.com/PaddlePaddle/PaddleX
// PaddleX's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.paddle.predictor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.core.MatManager;
import io.github.flux.core.RecognitionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.CTCLabelDecode;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PaddleRecognitionPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final List<ImageProcessor> preProcessors;
    private final CTCLabelDecode postProcessor;

    public PaddleRecognitionPredictor(final String modelFile,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final List<ImageProcessor> preProcessors,
                                      final CTCLabelDecode postProcessor) {
        try {
            this.postProcessor = postProcessor;
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);

            this.inputName = List.copyOf(session.getInputNames()).getFirst();

            this.preProcessors = List.copyOf(preProcessors);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private List<Mat> transform(List<Mat> data, MatManager matManager, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(matManager, arrays);
        }
        return arrays;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    public List<List<RecognitionResult>> batchPredict(List<Mat> images,
                                                      MatManager matManager, NDManager manager) throws OrtException {
        List<Mat> padedImages = ImageUtil.padImageToSame(matManager, images);
        List<Mat> transformedResults = transform(padedImages, matManager, preProcessors);
        try (
                OnnxTensor onnxInput = ImageUtil.matToOnnxTensor(transformedResults, env);
                OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput))
        ) {
            OnnxValue optinalResult = onnxResult.get(0);
            List<List<RecognitionResult>> allResults = new ArrayList<>();
            float[][][] v = (float[][][]) optinalResult.getValue();
            for (float[][] floats : v) {
                NDArray preds = manager.create(floats);

                List<RecognitionResult> recognitionResult = postProcessor.process(preds);
                allResults.add(recognitionResult);
                preds.close();
            }
            return allResults;
        } finally {
            // Release padded and transformed Mats that are no longer needed after inference
            for (Mat padded : padedImages) {
                matManager.release(padded);
            }
            for (Mat transformed : transformedResults) {
                matManager.release(transformed);
            }
        }
    }

}
