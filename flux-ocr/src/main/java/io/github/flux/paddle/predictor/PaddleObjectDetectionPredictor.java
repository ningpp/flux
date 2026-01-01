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
        List<List<ObjectDetectionResult>> results = new ArrayList<>();
        for (ProcessedMat mat : mats) {
            results.add(predict(mat, manager, true));
        }
        return results;
    }

    @Override
    public ProcessedMat processRgb(Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

    public List<ObjectDetectionResult> predict(String image, NDManager manager,
                                               boolean layoutNms) {
        Mat rgbImg = ImageUtil.readToRgb(image);
        List<ObjectDetectionResult> results = predict(
                new ProcessedMat(rgbImg.width(), rgbImg.height(), rgbImg),
                manager, layoutNms);
        rgbImg.release();
        return results;
    }

    public List<ObjectDetectionResult> predict(ProcessedMat image, NDManager manager,
                                               boolean layoutNms) {

        try {
            ObjectDetectionResizeResult resizeResult = detResize.process(image.processed());
            Mat transformedResult = transform(List.of(resizeResult.result_img()), preProcessors).get(0);

            int size = (int) (transformedResult.total() * transformedResult.channels());
            float[] floatDatas = new float[size];
            // Copy the data from Mat to the float array
            transformedResult.get(0, 0, floatDatas);

            FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
            long[] shape = new long[]{1, transformedResult.channels(), transformedResult.rows(), transformedResult.cols()};

            Map<String, OnnxTensor> inputs = new HashMap<>(3);
            inputs.put("image", OnnxTensor.createTensor(env, dataBuffer, shape));

            if (inputNames.size() == 3) {
                FloatBuffer im_shape_buffer = FloatBuffer.wrap(new float[]{
                        resizeResult.img_size()[0],
                        resizeResult.img_size()[1]
                });
                inputs.put("im_shape", OnnxTensor.createTensor(env, im_shape_buffer, new long[]{1, 2}));
            }

            // 特别注意这个值是反的
            FloatBuffer scale_factor_buffer = FloatBuffer.wrap(new float[]{
                    (float) resizeResult.scale_factors()[1],
                    (float) resizeResult.scale_factors()[0]
            });
            inputs.put("scale_factor", OnnxTensor.createTensor(env, scale_factor_buffer, new long[]{1, 2}));

            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            OnnxValue pred_boxes_value = onnxResult.get("fetch_name_0").get();
            float[][] pred_boxes = (float[][]) pred_boxes_value.getValue();

            NDArray boxes = manager.create(pred_boxes);

            List<ObjectDetectionResult> postProcessResult = postProcessor.process(boxes,
                    new long[]{resizeResult.ori_img_size()[0], resizeResult.ori_img_size()[1]},
                    this.threshold,
                    layoutNms,
                    null,
                    null
            );

            boxes.close();
            OnnxUtil.closeTensors(inputs);
            onnxResult.close();
            transformedResult.release();

            return postProcessResult;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

}
