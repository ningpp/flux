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
import ai.onnxruntime.OrtSession;
import io.github.flux.core.RecognitionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.CTCLabelDecode;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class PaddleRecognitionPredictor implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
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

    private List<Mat> transform(List<Mat> data, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(arrays);
        }
        return arrays;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    public List<List<RecognitionResult>> batchPredict(List<Mat> images, NDManager manager) {
        try {
            List<Mat> padedImages = padImageToSame(images);
            List<Mat> transformedResults = transform(padedImages, preProcessors);

            int height = transformedResults.get(0).rows();
            int width = transformedResults.get(0).cols();
            int channels = transformedResults.get(0).channels();
            int oneSize = (int) (transformedResults.get(0).total() * channels);
            int size = transformedResults.size() * oneSize;
            float[] floatDatas = new float[size];
            int index = 0;
            for (Mat pad : transformedResults) {
                float[] oneDatas = new float[oneSize];
                pad.get(0, 0, oneDatas);
                System.arraycopy(oneDatas, 0, floatDatas, index, oneDatas.length);
                index += oneSize;
            }

            for (Mat mat : images) {
                mat.release();
            }

            for (Mat mat : transformedResults) {
                mat.release();
            }

            for (Mat mat : padedImages) {
                mat.release();
            }

            FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
            long[] shape = new long[] {
                    transformedResults.size(),
                    channels,
                    height,
                    width
            };
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
            for (String inputName : inputNames) {
                inputs.put(inputName, onnxInput);
            }

            OrtSession.Result onnxResult = session.run(inputs, outputNames);

            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            List<List<RecognitionResult>> allResults = new ArrayList<>();
            if (optinalResult.isPresent()) {
                float[][][] v =  (float[][][]) optinalResult.get().getValue();
                for (float[][] floats : v) {
                    NDArray preds = manager.create(floats);

                    List<RecognitionResult> recognitionResult = postProcessor.process(preds);
                    allResults.add(recognitionResult);
                    preds.close();
                }
            } else {
                throw new FluxException("未知错误！");
            }
            onnxResult.close();
            onnxInput.close();
            return allResults;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public static List<Mat> padImageToSame(List<Mat> images) {
        int maxWidth = 0;
        int maxHeight = 0;

        // Load images and find maximum dimensions
        for (Mat image : images) {
            maxWidth = Math.max(maxWidth, image.cols());
            maxHeight = Math.max(maxHeight, image.rows());
        }

        Scalar paddingColor = new Scalar(255, 255, 255);
        List<Mat> results = new ArrayList<>(images.size());
        for (Mat image : images) {
            Mat pad = ImageUtil.padImageToSize(image, maxWidth, maxHeight, paddingColor);
            results.add(pad);
        }
        return results;
    }

}
