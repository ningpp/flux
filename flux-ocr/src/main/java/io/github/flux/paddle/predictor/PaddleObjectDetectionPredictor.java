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
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResizeResult;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.DetPostProcessor;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.ObjectDetectionResize;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.OnnxUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PaddleObjectDetectionPredictor extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    private final String modelName;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
    private final ObjectDetectionResize detResize;
    private final List<ImageProcessor> preProcessors;
    private final DetPostProcessor postProcessor;
    private final Object threshold;

    public PaddleObjectDetectionPredictor(final String modelRootDir,
                                          final String modelName,
                                          final int gpuIndex,
                                          final OrtEnvironment env,
                                          final Object threshold,
                                          final ObjectDetectionResize detResize,
                                          final List<ImageProcessor> preProcessors,
                                          final DetPostProcessor detPostProcessor) {
        try {
            this.threshold = threshold;
            this.postProcessor = detPostProcessor;
            this.detResize = detResize;
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.modelName = modelName;
            String modelDir = modelRootDir + File.separator + modelName;
            String modelFile = new File(modelDir, "model.onnx").getAbsolutePath();
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();

            this.preProcessors = List.copyOf(preProcessors);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private List<Mat> transform(List<Mat> data, MatManager matManager, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(matManager, arrays);
        }
        return arrays;
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predict(mats, matManager, manager, true);
    }

    @Override
    public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

    public List<ObjectDetectionResult> predict(String image, MatManager matManager, NDManager manager,
                                               boolean layoutNms) {
        Mat rgbImg = ImageUtil.readToRgb(matManager, image);
        List<ObjectDetectionResult> results = predict(
                List.of(new ProcessedMat(rgbImg.width(), rgbImg.height(), rgbImg)),
                matManager, manager, layoutNms).get(0);
        matManager.release(rgbImg);
        return results;
    }

    public List<List<ObjectDetectionResult>> predict(List<ProcessedMat> images, MatManager matManager, NDManager manager,
                                                     boolean layoutNms) {
        if ("PP-DocLayoutV2".equals(modelName)) {
            return _predict(images, matManager, manager, layoutNms);
            /*
            List<List<ObjectDetectionResult>> results = new ArrayList<>();
            for (var pm : images) {
                results.addAll(_predict(List.of(pm), matManager, manager, layoutNms));
            }
            return results;
            */
        } else {
            return _predict(images, matManager, manager, layoutNms);
        }
    }

    public List<List<ObjectDetectionResult>> _predict(List<ProcessedMat> images, MatManager matManager, NDManager manager,
                                                      boolean layoutNms) {

        Map<String, OnnxTensor> inputs = new HashMap<>(3);
        OrtSession.Result onnxResult = null;
        try {
            List<ObjectDetectionResizeResult> resizeResults = new ArrayList<>();
            float[] scale_factors = new float[images.size() * 2];
            float[] im_shapes = new float[images.size() * 2];
            int channels = 0;
            int rows = 0;
            int cols = 0;
            float[][] allFloatDatas = new float[images.size()][];
            for (int i = 0; i < images.size(); i++) {
                ObjectDetectionResizeResult resizeResult = detResize.process(matManager, images.get(i).processed());
                resizeResults.add(resizeResult);
                Mat transformedResult = transform(List.of(resizeResult.result_img()), matManager, preProcessors).get(0);

                scale_factors[i * 2] = (float) resizeResult.scale_factors()[1];
                scale_factors[i * 2 + 1] = (float) resizeResult.scale_factors()[0];

                im_shapes[i * 2] = (float) resizeResult.img_size()[0];
                im_shapes[i * 2 + 1] = (float) resizeResult.img_size()[1];

                channels = transformedResult.channels();
                rows = transformedResult.rows();
                cols = transformedResult.cols();

                float[] floatDatas = new float[channels * rows * cols];
                // Copy the data from Mat to the float array
                transformedResult.get(0, 0, floatDatas);
                IOUtil.close(transformedResult);
                allFloatDatas[i] = floatDatas;
            }

            FloatBuffer dataBuffer = FloatBuffer.wrap(ArrayUtil.flat(allFloatDatas));
            long[] shape = new long[] { images.size(), channels, rows, cols };

            inputs.put("image", OnnxTensor.createTensor(env, dataBuffer, shape));

            if (inputNames.size() == 3) {
                FloatBuffer im_shape_buffer = FloatBuffer.wrap(im_shapes);
                inputs.put("im_shape", OnnxTensor.createTensor(env, im_shape_buffer, new long[]{images.size(), 2}));
            }

            // 特别注意这个值是反的
            FloatBuffer scale_factor_buffer = FloatBuffer.wrap(scale_factors);
            inputs.put("scale_factor", OnnxTensor.createTensor(env, scale_factor_buffer, new long[]{images.size(), 2}));

            onnxResult = session.run(inputs, outputNames);
            OnnxValue pred_boxes_value = onnxResult.get("fetch_name_0").get();
            float[][] pred_boxes = (float[][]) pred_boxes_value.getValue();

            NDArray boxes = manager.create(pred_boxes);
            long[] inferResultShape = boxes.getShape().getShape();
            NDArray reshaped = boxes.reshape(images.size(), inferResultShape[0]/images.size(), inferResultShape[1]);
            NDArray[] results = new NDArray[images.size()];
            for (int i = 0; i < images.size(); i++) {
                results[i] = reshaped.get(i);
            }

            List<List<ObjectDetectionResult>> postProcessResults = new ArrayList<>();
            for (int i = 0; i < images.size(); i++) {
                postProcessResults.add(postProcessor.process(results[i],
                        new long[]{
                                resizeResults.get(i).ori_img_size()[0],
                                resizeResults.get(i).ori_img_size()[1]
                        },
                        this.threshold,
                        layoutNms,
                        null,
                        null
                ));
                IOUtil.close(results[i]);
            }

            IOUtil.close(reshaped);
            boxes.close();

            return postProcessResults;
        } catch (Exception e) {
            throw new FluxException(e);
        } finally {
            OnnxUtil.closeTensors(inputs);
            IOUtil.close(onnxResult);
        }
    }

}
