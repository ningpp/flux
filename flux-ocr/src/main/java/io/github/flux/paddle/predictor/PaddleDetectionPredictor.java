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
import io.github.flux.util.OnnxSessionUtil;
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

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PaddleDetectionPredictor implements AutoCloseable {

    public record StageTimings(long padNanos,
                               long resizeNanos,
                               long preprocessNanos,
                               long tensorCreateNanos,
                               long inferenceNanos,
                               long outputReadNanos,
                               long postprocessNanos,
                               long cleanupNanos,
                               long totalNanos) {
    }

    public record TimedResult(List<TextDetectionResult> results, StageTimings timings) {
    }

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

    public List<TextDetectionResult> batchPredict(List<Mat> images,
                                       MatManager matManager,
                                       NDManager manager,
                                       final Integer limitSideLen,
                                       final LimitType limitType,
                                       final Integer maxSideLimit,
                                       final Float thresh,
                                       final Float boxThresh,
                                       final Float unclipRatio) throws OrtException {
        return batchPredictWithTimings(
                images,
                matManager,
                manager,
                limitSideLen,
                limitType,
                maxSideLimit,
                thresh,
                boxThresh,
                unclipRatio).results();
    }

    public TimedResult batchPredictWithTimings(List<Mat> images,
                                               MatManager matManager,
                                               NDManager manager,
                                               final Integer limitSideLen,
                                               final LimitType limitType,
                                               final Integer maxSideLimit,
                                               final Float thresh,
                                               final Float boxThresh,
                                               final Float unclipRatio) throws OrtException {
        return batchPredictInternal(images, matManager, manager, limitSideLen, limitType, maxSideLimit,
                thresh, boxThresh, unclipRatio, true);
    }

    TimedResult batchPredictWithLegacyTimings(List<Mat> images,
                                              MatManager matManager,
                                              NDManager manager,
                                              final Integer limitSideLen,
                                              final LimitType limitType,
                                              final Integer maxSideLimit,
                                              final Float thresh,
                                              final Float boxThresh,
                                              final Float unclipRatio) throws OrtException {
        return batchPredictInternal(images, matManager, manager, limitSideLen, limitType, maxSideLimit,
                thresh, boxThresh, unclipRatio, false);
    }

    private TimedResult batchPredictInternal(List<Mat> images,
                                             MatManager matManager,
                                             NDManager manager,
                                             final Integer limitSideLen,
                                             final LimitType limitType,
                                             final Integer maxSideLimit,
                                             final Float thresh,
                                             final Float boxThresh,
                                             final Float unclipRatio,
                                             final boolean directPostProcess) throws OrtException {
        long totalStart = System.nanoTime();

        long padStart = System.nanoTime();
        List<Mat> sameSizeMats = ImageUtil.padImageToSame(matManager, images);
        long padNanos = System.nanoTime() - padStart;

        List<Pair<DetResizeResultV2, Mat>> detResizeResults = new ArrayList<>();
        long resizeNanos = 0L;
        long preprocessNanos = 0L;
        for (Mat sameSizeMat : sameSizeMats) {
            long resizeStart = System.nanoTime();
            DetResizeResultV2 resizedResult = detResize.process(matManager, List.of(sameSizeMat),
                    limitSideLen, limitType, maxSideLimit).get(0);
            resizeNanos += System.nanoTime() - resizeStart;

            long preprocessStart = System.nanoTime();
            Mat preprocessed = transform(List.of(resizedResult.resizeImg()), matManager, preProcessors).get(0);
            preprocessNanos += System.nanoTime() - preprocessStart;
            detResizeResults.add(Pair.of(resizedResult, preprocessed));
        }
        OnnxTensor onnxInput = null;
        OrtSession.Result onnxResult = null;
        List<TextDetectionResult> tdResults = new ArrayList<>();
        long tensorCreateNanos = 0L;
        long inferenceNanos = 0L;
        long outputReadNanos = 0L;
        long postprocessNanos = 0L;
        long cleanupNanos;
        try {
            long tensorCreateStart = System.nanoTime();
            onnxInput = matManager.track(ImageUtil.matToOnnxTensor(
                    detResizeResults.stream().map(Pair::getRight).toList(), env));
            tensorCreateNanos = System.nanoTime() - tensorCreateStart;

            long inferenceStart = System.nanoTime();
            onnxResult = matManager.runSession(session, Map.of(inputName, onnxInput));
            inferenceNanos = System.nanoTime() - inferenceStart;

            long outputReadStart = System.nanoTime();
            OnnxValue outputResult = onnxResult.get(0);
            if (directPostProcess) {
                OnnxTensor outputTensor = (OnnxTensor) outputResult;
                long[] outputShape = outputTensor.getInfo().getShape();
                if (outputShape.length != 4) {
                    throw new FluxException("Unexpected detection output shape length: " + outputShape.length);
                }
                int batchSize = Math.toIntExact(outputShape[0]);
                int channelCount = Math.toIntExact(outputShape[1]);
                int outputHeight = Math.toIntExact(outputShape[2]);
                int outputWidth = Math.toIntExact(outputShape[3]);
                if (channelCount != 1) {
                    throw new FluxException("Unexpected detection output channel count: " + channelCount);
                }
                FloatBuffer outputBuffer = outputTensor.getFloatBuffer();
                outputReadNanos = System.nanoTime() - outputReadStart;

                long postprocessStart = System.nanoTime();
                float resolvedThresh = thresh == null ? dbPostProcess.getThresh() : thresh;
                float resolvedBoxThresh = boxThresh == null ? dbPostProcess.getBoxThresh() : boxThresh;
                float resolvedUnclipRatio = unclipRatio == null ? dbPostProcess.getUnclipRatio() : unclipRatio;
                int oneResultSize = channelCount * outputHeight * outputWidth;
                for (int b = 0; b < batchSize; b++) {
                    float[] predFlat = readOutputPlane(outputBuffer, b * oneResultSize, outputHeight, outputWidth);
                    double[] imageShape = detResizeResults.get(b).getLeft().imgShape();
                    tdResults.add(dbPostProcess.boxesFromBitmap(
                            matManager,
                            predFlat,
                            outputHeight,
                            outputWidth,
                            Double.valueOf(imageShape[1]).intValue(),
                            Double.valueOf(imageShape[0]).intValue(),
                            resolvedThresh,
                            resolvedBoxThresh,
                            resolvedUnclipRatio));
                }
                postprocessNanos = System.nanoTime() - postprocessStart;
            } else {
                float[][][][] data = (float[][][][]) outputResult.getValue();
                outputReadNanos = System.nanoTime() - outputReadStart;

                long postprocessStart = System.nanoTime();
                appendLegacyPostProcessResults(
                        tdResults,
                        data,
                        matManager,
                        manager,
                        detResizeResults,
                        thresh,
                        boxThresh,
                        unclipRatio);
                postprocessNanos = System.nanoTime() - postprocessStart;
            }
        } finally {
            long cleanupStart = System.nanoTime();
            matManager.release(onnxResult);
            matManager.release(onnxInput);
            // 释放检测预处理阶段产生的 Mat，避免长期存活的 MatManager 累积原生内存。
            // sameSizeMats 可能已被 detResize 内部释放，pair.getRight()（预处理结果）在
            // matToOnnxTensor 拷贝数据后不再需要；matManager.release 幂等，重复释放安全。
            for (Mat sameSizeMat : sameSizeMats) {
                matManager.release(sameSizeMat);
            }
            for (Pair<DetResizeResultV2, Mat> pair : detResizeResults) {
                matManager.release(pair.getRight());
            }
            cleanupNanos = System.nanoTime() - cleanupStart;
        }
        long totalNanos = System.nanoTime() - totalStart;
        return new TimedResult(
                tdResults,
                new StageTimings(
                        padNanos,
                        resizeNanos,
                        preprocessNanos,
                        tensorCreateNanos,
                        inferenceNanos,
                        outputReadNanos,
                        postprocessNanos,
                        cleanupNanos,
                        totalNanos));
    }

    private float[] readOutputPlane(FloatBuffer outputBuffer, int offset, int height, int width) {
        float[] predFlat = new float[height * width];
        FloatBuffer batchBuffer = outputBuffer.duplicate();
        batchBuffer.position(offset);
        batchBuffer.get(predFlat);
        return predFlat;
    }

    private void appendLegacyPostProcessResults(List<TextDetectionResult> tdResults,
                                                float[][][][] data,
                                                MatManager matManager,
                                                NDManager manager,
                                                List<Pair<DetResizeResultV2, Mat>> detResizeResults,
                                                Float thresh,
                                                Float boxThresh,
                                                Float unclipRatio) {
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
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

}
