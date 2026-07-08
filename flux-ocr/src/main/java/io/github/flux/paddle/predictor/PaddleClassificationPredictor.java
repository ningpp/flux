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
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.util.OnnxSessionUtil;
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
    // 复用单块临时 buffer（约 3*H*W 个 float），避免每帧重新分配，降低 GC 压力与峰值内存。
    // 单个 predictor 对应固定输入尺寸，且 OrtSession 非线程安全（不并发调用），故可安全复用。
    private float[] oneDatas;

    public PaddleClassificationPredictor(final String modelFile,
                                         final int gpuIndex,
                                         final OrtEnvironment env,
                                         final List<ImageProcessor> preProcessors,
                                         final List<String> labels) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);

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

    public List<List<ClassificationResult>> doBatchPredict(List<PreProcessResult> pprs,
                                                           Map<String, Object> extraParameters) {
        try {
            List<List<ClassificationResult>> allResults = new ArrayList<>();
            Integer k = ParameterUtil.getInteger(extraParameters, "k");
            if (k == null) {
                k = 1;
            }

            try (
                    OnnxTensor onnxInput = toOnnxTensor(pprs);
                    OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput));
            ) {
                OnnxValue optinalResult = onnxResult.get(0);
                float[][] preditResult = (float[][]) optinalResult.getValue();
                for (int i = 0; i < pprs.size(); i++) {
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

    /**
     * 将预处理后的 Mat 列表拼接为 NCHW 的 OnnxTensor。
     * 复用实例级 {@link #oneDatas} 临时 buffer（单张尺寸固定），避免每帧重新分配
     * 约 3*H*W 个 float（文档方向模型约 602KB），降低 GC 压力与峰值内存。
     */
    private OnnxTensor toOnnxTensor(List<PreProcessResult> pprs) {
        List<Mat> mats = PreProcessResult.getMats(pprs);
        int height = mats.get(0).rows();
        int width = mats.get(0).cols();
        int channels = mats.get(0).channels();
        int oneSize = (int) (mats.get(0).total() * channels);
        int size = mats.size() * oneSize;

        float[] floatDatas = new float[size];
        if (oneDatas == null || oneDatas.length != oneSize) {
            oneDatas = new float[oneSize];
        }
        int index = 0;
        for (Mat pad : mats) {
            pad.get(0, 0, oneDatas);
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
        try {
            return OnnxTensor.createTensor(env, dataBuffer, shape);
        } catch (OrtException e) {
            throw new FluxException(e);
        }
    }

}
