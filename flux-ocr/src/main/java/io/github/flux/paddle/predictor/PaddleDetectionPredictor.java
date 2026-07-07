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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.DBPostProcess;
import io.github.flux.paddle.processor.DetResize;
import io.github.flux.paddle.processor.DetResize.DetResizeResultV2;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.LimitType;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.ImageUtil;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PaddleDetectionPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final DetResize detResize;
    private final List<ImageProcessor> preProcessors;
    private final DBPostProcess dbPostProcess;

    public PaddleDetectionPredictor(final String modelFile,
                                    final int gpuIndex,
                                    final OrtEnvironment env,
                                    final DetResize detResize,
                                    final List<ImageProcessor> preProcessors,
                                    final DBPostProcess dbPostProcess) {
        try {
            this.dbPostProcess = dbPostProcess;
            this.detResize = detResize;
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

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

    public List<TextDetectionResult> batchPredict(List<Mat> images,
                                       MatManager matManager,
                                       NDManager manager,
                                       final Integer limitSideLen,
                                       final LimitType limitType,
                                       final Integer maxSideLimit,
                                       final Float thresh,
                                       final Float boxThresh,
                                       final Float unclipRatio) throws OrtException {
        List<int[]> srcSizes = new ArrayList<>();
        for (Mat mat : images) {
            srcSizes.add(new int[] {mat.rows(), mat.cols()});
        }
        List<Mat> sameSizeMats = ImageUtil.padImageToSame(matManager, images);
        List<Pair<DetResizeResultV2, Mat>> detResizeResults = new ArrayList<>();
        for (Mat sameSizeMat : sameSizeMats) {
            DetResizeResultV2 resizedResult = detResize.process(matManager, List.of(sameSizeMat),
                    limitSideLen, limitType, maxSideLimit).get(0);
            Mat preprocessed = transform(List.of(resizedResult.resizeImg()), matManager, preProcessors).get(0);
            detResizeResults.add(Pair.of(resizedResult, preprocessed));
        }
        try (
                OnnxTensor onnxInput = ImageUtil.matToOnnxTensor(
                        detResizeResults.stream().map(Pair::getRight).toList(), env);
                OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput))
        ) {
            OnnxValue outputResult = onnxResult.get(0);
            float[][][][] data = (float[][][][]) outputResult.getValue();
            List<TextDetectionResult> tdResults = new ArrayList<>();
            for (int b = 0; b < data.length; b++) {
                NDList preds = new NDList();
                List<NDArray> postBoxes = List.of();
                try {
                    preds.add(ArrayUtil.toNDArray(manager, data[b]));

                    Pair<List<NDArray>, List<Float>> postResult = dbPostProcess.call(
                            matManager,
                            preds,
                            detResizeResults.get(b).getLeft().imgShape(),
                            thresh == null ? dbPostProcess.getThresh() : thresh,
                            boxThresh == null ? dbPostProcess.getBoxThresh() : boxThresh,
                            unclipRatio == null ? dbPostProcess.getUnclipRatio() : unclipRatio
                    );

                    postBoxes = postResult.getKey();
                    int[][][] polys = new int[postBoxes.size()][][];
                    int index = 0;
                    for (NDArray array : postBoxes) {
                        long[] arrayShape = array.getShape().getShape();
                        int rows = (int) arrayShape[0];
                        int cols = (int) arrayShape[1];
                        int[][] result = new int[rows][cols];
                        for (int i = 0; i < rows; i++) {
                            for (int j = 0; j < cols; j++) {
                                result[i][j] = array.getInt(i, j);
                            }
                        }
                        polys[index] = result;
                        index++;
                    }
                    tdResults.add(new TextDetectionResult(polys, postResult.getValue()));
                } finally {
                    for (NDArray array : postBoxes) {
                        array.close();
                    }
                    preds.close();
                }
            }
            return tdResults;
        } finally {
            // 释放检测预处理阶段产生的 Mat，避免长期存活的 MatManager 累积原生内存。
            // sameSizeMats 可能已被 detResize 内部释放，pair.getRight()（预处理结果）在
            // matToOnnxTensor 拷贝数据后不再需要；matManager.release 幂等，重复释放安全。
            for (Mat sameSizeMat : sameSizeMats) {
                matManager.release(sameSizeMat);
            }
            for (Pair<DetResizeResultV2, Mat> pair : detResizeResults) {
                matManager.release(pair.getRight());
            }
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

}
