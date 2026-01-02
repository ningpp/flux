package io.github.flux.paddle.predictor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.BatchPredictor;
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

    private List<Mat> transform(List<Mat> data, List<ImageProcessor> processors) {
        List<Mat> arrays = data;
        for (ImageProcessor processor : processors) {
            arrays = processor.process(arrays);
        }
        return arrays;
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, NDManager manager, Map<String, Object> extraParameters) {
        return predict(mats, manager, true);
    }

    @Override
    public ProcessedMat processRgb(Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

    public List<ObjectDetectionResult> predict(String image, NDManager manager,
                                               boolean layoutNms) {
        Mat rgbImg = ImageUtil.readToRgb(image);
        List<ObjectDetectionResult> results = predict(
                List.of(new ProcessedMat(rgbImg.width(), rgbImg.height(), rgbImg)),
                manager, layoutNms).get(0);
        rgbImg.release();
        return results;
    }

    public List<List<ObjectDetectionResult>> predict(List<ProcessedMat> images, NDManager manager,
                                                     boolean layoutNms) {
        if ("PP-DocLayoutV2".equals(modelName)) {
            List<List<ObjectDetectionResult>> results = new ArrayList<>();
            for (var pm : images) {
                results.addAll(_predict(List.of(pm), manager, layoutNms));
            }
            return results;
        } else {
            return _predict(images, manager, layoutNms);
        }
    }

    public List<List<ObjectDetectionResult>> _predict(List<ProcessedMat> images, NDManager manager,
                                                      boolean layoutNms) {

        try {
            List<ObjectDetectionResizeResult> resizeResults = new ArrayList<>();
            float[] scale_factors = new float[images.size() * 2];
            float[] im_shapes = new float[images.size() * 2];
            int channels = 0;
            int rows = 0;
            int cols = 0;
            float[][] allFloatDatas = new float[images.size()][];
            for (int i = 0; i < images.size(); i++) {
                ObjectDetectionResizeResult resizeResult = detResize.process(images.get(i).processed());
                resizeResults.add(resizeResult);
                Mat transformedResult = transform(List.of(resizeResult.result_img()), preProcessors).get(0);

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

            Map<String, OnnxTensor> inputs = new HashMap<>(3);
            inputs.put("image", OnnxTensor.createTensor(env, dataBuffer, shape));

            if (inputNames.size() == 3) {
                FloatBuffer im_shape_buffer = FloatBuffer.wrap(im_shapes);
                inputs.put("im_shape", OnnxTensor.createTensor(env, im_shape_buffer, new long[]{images.size(), 2}));
            }

            // 特别注意这个值是反的
            FloatBuffer scale_factor_buffer = FloatBuffer.wrap(scale_factors);
            inputs.put("scale_factor", OnnxTensor.createTensor(env, scale_factor_buffer, new long[]{images.size(), 2}));

            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            OnnxValue pred_boxes_value = onnxResult.get("fetch_name_0").get();
            float[][] pred_boxes = (float[][]) pred_boxes_value.getValue();

            NDArray boxes = manager.create(pred_boxes);

            List<List<ObjectDetectionResult>> postProcessResults = new ArrayList<>();
            for (int i = 0; i < images.size(); i++) {
                postProcessResults.add(postProcessor.process(boxes,
                        new long[]{
                                resizeResults.get(i).ori_img_size()[0],
                                resizeResults.get(i).ori_img_size()[1]
                        },
                        this.threshold,
                        layoutNms,
                        null,
                        null
                ));
            }

            boxes.close();
            OnnxUtil.closeTensors(inputs);
            onnxResult.close();

            return postProcessResults;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

}
