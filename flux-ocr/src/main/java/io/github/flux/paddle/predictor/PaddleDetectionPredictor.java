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
import ai.djl.util.Pair;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
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
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class PaddleDetectionPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
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

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();

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

    public TextDetectionResult predict(Mat image,
                                       MatManager matManager,
                                       NDManager manager,
                                       final Integer limitSideLen,
                                       final LimitType limitType,
                                       final Integer maxSideLimit,
                                       final Float thresh,
                                       final Float boxThresh,
                                       final Float unclipRatio) {
        try {
            DetResizeResultV2 resizedResult = detResize.process(matManager, List.of(image),
                    limitSideLen, limitType, maxSideLimit).get(0);
            double[] resizedImageShapes = resizedResult.imgShape();
            Mat transformedResult = transform(List.of(resizedResult.resizeImg()), matManager, preProcessors).get(0);
            resizedResult.resizeImg().release();

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
                Object v = optinalResult.get().getValue();
                NDList preds = new NDList();
                float[][][][] data = (float[][][][]) v;
                preds.add(ArrayUtil.toNDArray(manager, data[0]));

                Pair<List<NDArray>, List<Float>> postResult = dbPostProcess.call(
                        matManager,
                        preds,
                        resizedImageShapes,
                        thresh == null ? dbPostProcess.getThresh() : thresh,
                        boxThresh == null ? dbPostProcess.getBoxThresh() : boxThresh,
                        unclipRatio == null ? dbPostProcess.getUnclipRatio() : unclipRatio
                );

                int[][][] polys = new int[postResult.getKey().size()][][];
                int index = 0;
                for (NDArray array : postResult.getKey()) {
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
                    array.close();
                    index++;
                }
                /**/
                preds.close();
                onnxResult.close();
                onnxInput.close();
                transformedResult.release();
                image.release();
                return new TextDetectionResult(polys, postResult.getValue());
            }
            image.release();
            onnxResult.close();
            onnxInput.close();
            transformedResult.release();
        } catch (Exception e) {
            throw new FluxException(e);
        }
        throw new FluxException("未知错误！");
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

}
