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

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PaddleRecognitionPredictor implements AutoCloseable {

    public record StageTimings(long padNanos,
                               long preprocessNanos,
                               long tensorCreateNanos,
                               long inferenceNanos,
                               long outputReadNanos,
                               long postprocessNanos,
                               long cleanupNanos,
                               long totalNanos) {
    }

    public record TimedResult(List<List<RecognitionResult>> results, StageTimings timings) {
    }

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
        return batchPredictWithTimings(images, matManager, manager).results();
    }

    public TimedResult batchPredictWithTimings(List<Mat> images,
                                               MatManager matManager,
                                               NDManager manager) throws OrtException {
        long totalStart = System.nanoTime();

        long padStart = System.nanoTime();
        List<Mat> padedImages = ImageUtil.padImageToSame(matManager, images);
        long padNanos = System.nanoTime() - padStart;

        long preprocessStart = System.nanoTime();
        List<Mat> transformedResults = transform(padedImages, matManager, preProcessors);
        long preprocessNanos = System.nanoTime() - preprocessStart;

        OnnxTensor onnxInput = null;
        OrtSession.Result onnxResult = null;
        List<List<RecognitionResult>> allResults = new ArrayList<>();
        long tensorCreateNanos = 0L;
        long inferenceNanos = 0L;
        long outputReadNanos = 0L;
        long postprocessNanos = 0L;
        long cleanupNanos;
        try {
            long tensorCreateStart = System.nanoTime();
            onnxInput = matManager.track(ImageUtil.matToOnnxTensor(transformedResults, env));
            tensorCreateNanos = System.nanoTime() - tensorCreateStart;

            long inferenceStart = System.nanoTime();
            onnxResult = matManager.runSession(session, Map.of(inputName, onnxInput));
            inferenceNanos = System.nanoTime() - inferenceStart;

            long outputReadStart = System.nanoTime();
            OnnxTensor outputTensor = (OnnxTensor) onnxResult.get(0);
            long[] outputShape = outputTensor.getInfo().getShape();
            if (outputShape.length != 3) {
                throw new FluxException("Unexpected recognition output shape length: " + outputShape.length);
            }
            int batchSize = Math.toIntExact(outputShape[0]);
            int sequenceLength = Math.toIntExact(outputShape[1]);
            int classCount = Math.toIntExact(outputShape[2]);
            FloatBuffer outputBuffer = outputTensor.getFloatBuffer();
            outputReadNanos = System.nanoTime() - outputReadStart;

            long postprocessStart = System.nanoTime();
            int oneResultSize = sequenceLength * classCount;
            for (int batch = 0; batch < batchSize; batch++) {
                allResults.add(postProcessor.process(
                        outputBuffer,
                        batch * oneResultSize,
                        sequenceLength,
                        classCount));
            }
            postprocessNanos = System.nanoTime() - postprocessStart;
        } finally {
            long cleanupStart = System.nanoTime();
            matManager.release(onnxResult);
            matManager.release(onnxInput);
            // Release padded and transformed Mats that are no longer needed after inference
            for (Mat padded : padedImages) {
                matManager.release(padded);
            }
            for (Mat transformed : transformedResults) {
                matManager.release(transformed);
            }
            cleanupNanos = System.nanoTime() - cleanupStart;
        }
        long totalNanos = System.nanoTime() - totalStart;
        return new TimedResult(
                allResults,
                new StageTimings(
                        padNanos,
                        preprocessNanos,
                        tensorCreateNanos,
                        inferenceNanos,
                        outputReadNanos,
                        postprocessNanos,
                        cleanupNanos,
                        totalNanos));
    }

}
