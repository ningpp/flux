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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.ClassificationResult;
import io.github.flux.paddle.processor.TopkProcessor;
import io.github.flux.core.TopkResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ImageProcessor;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class PaddleClassificationPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
    private final List<ImageProcessor> preProcessors;
    private final TopkProcessor topkProcessor;

    public PaddleClassificationPredictor(final String modelFile,
                                         final int gpuIndex,
                                         final OrtEnvironment env,
                                         final List<ImageProcessor> preProcessors,
                                         final List<String> labels) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();

            this.preProcessors = List.copyOf(preProcessors);
            this.topkProcessor = new TopkProcessor(List.copyOf(labels));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private List<Mat> transform(List<Mat> data, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(arrays);
        }
        return arrays;
    }

    public Mat process(Mat mat) {
        for (ImageProcessor processor : preProcessors) {
            mat = processor.process(mat);
        }
        return mat;
    }

    public List<ClassificationResult> predict(Mat image, final int k) {
        Mat transformedResult = transform(List.of(image), preProcessors).get(0);
        return predictProcessed(transformedResult, k);
    }

    public List<ClassificationResult> predictProcessed(Mat transformedResult, final int k) {
        try {
            int size = (int) (transformedResult.total() * transformedResult.channels());
            float[] floatDatas = new float[size];
            // Copy the data from Mat to the float array
            transformedResult.get(0, 0, floatDatas);

            FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
            long[] shape = new long[] {1, transformedResult.channels(), transformedResult.rows(), transformedResult.cols() };
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
            for (String inputName : inputNames) {
                inputs.put(inputName, onnxInput);
            }
            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            if (optinalResult.isPresent()) {
                float[][] preditResult = (float[][]) optinalResult.get().getValue();
                TopkResult topkResult = topkProcessor.compute(preditResult, k);
                List<ClassificationResult> results = new ArrayList<>(k);
                for (int i = 0; i < k; i++) {
                    results.add(new ClassificationResult(
                            topkResult.scores()[0][i],
                            topkResult.labels()[0][i]
                    ));
                }
                onnxResult.close();
                onnxInput.close();
                transformedResult.release();
                return results;
            }
            onnxResult.close();
            onnxInput.close();
            transformedResult.release();
        } catch (Exception e) {
            throw new FluxException(e);
        }
        throw new FluxException("未知错误！");
    }

}
