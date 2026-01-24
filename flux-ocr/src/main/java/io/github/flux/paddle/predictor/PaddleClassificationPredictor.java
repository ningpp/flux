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

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TopkResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.TopkProcessor;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PaddleClassificationPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
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

            this.inputName = List.copyOf(session.getInputNames()).getFirst();

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

    private List<Mat> transform(MatManager matManager, List<Mat> data, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(matManager, arrays);
        }
        return arrays;
    }

    public Mat process(MatManager matManager, Mat mat) {
        for (ImageProcessor processor : preProcessors) {
            mat = processor.process(matManager, mat);
        }
        return mat;
    }

    public List<List<ClassificationResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                           NDManager manager, Map<String, Object> extraParameters) {
        try {
            List<List<ClassificationResult>> allResults = new ArrayList<>();
            Integer k = ParameterUtil.getInteger(extraParameters, "k");
            if (k == null) {
                k = 1;
            }

            int height = mats.get(0).mat().rows();
            int width = mats.get(0).mat().cols();
            int channels = mats.get(0).mat().channels();
            int oneSize = (int) (mats.get(0).mat().total() * channels);
            int size = mats.size() * oneSize;
            float[] floatDatas = new float[size];
            int index = 0;
            for (PreProcessResult ppr : mats) {
                float[] oneDatas = new float[oneSize];
                ppr.mat().get(0, 0, oneDatas);
                System.arraycopy(oneDatas, 0, floatDatas, index, oneDatas.length);
                index += oneSize;
            }
            FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
            long[] shape = new long[] {
                    mats.size(),
                    channels,
                    height,
                    width
            };
            try (
                    OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
                    OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput));
            ) {
                OnnxValue optinalResult = onnxResult.get(0);
                float[][] preditResult = (float[][]) optinalResult.getValue();
                for (int i = 0; i < mats.size(); i++) {
                    float[][] softmax = new float[][] {preditResult[i]};
                    TopkResult topkResult = topkProcessor.compute(softmax, k);
                    List<ClassificationResult> results = new ArrayList<>();
                    for (int j = 0; j < k; j++) {
                        results.add(new ClassificationResult(
                                topkResult.scores()[0][j],
                                topkResult.labels()[0][j]
                        ));
                    }
                    allResults.add(results);
                }
                return allResults;
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

}
