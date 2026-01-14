// this code is convert from  https://github.com/docling-project/docling
// docling IS Licensed under the MIT License
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
package io.github.flux.docling;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Rescale;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.RtdetrPostProcessor;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ArrayUtil;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class DoclingLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;

    @Override
    public void close() throws Exception {
        session.close();
    }

    public DoclingLayoutModel(final String modelDir,
                              final String modelName,
                              final int gpuIndex,
                              final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            String modelFile = new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath();
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public static Set<String> MODEL_NAMES = Set.of(
            "docling-layout-egret-large",
            "docling-layout-egret-medium",
            "docling-layout-egret-xlarge",
            "docling-layout-heron",
            "docling-layout-heron-101"
    );

    public List<List<ObjectDetectionResult>> batchPredictFiles(List<String> images, MatManager matManager, NDManager manager, float threshold) {
        List<ProcessedMat> mats = new ArrayList<>();
        for (String image : images) {
            ProcessedMat rgbImg = process(matManager, image, manager);
            mats.add(rgbImg);
        }
        return batchPredict(mats, matManager, manager, threshold);
    }

    private static final List<ImageProcessor> PREPROCESSORS = List.of(
            new Resize(640, 640, Imgproc.INTER_AREA),
            new Rescale(0.00392156862745098),
            new ToCHWImage()
    );

    private static final RtdetrPostProcessor POST_PROCESSOR = new RtdetrPostProcessor(List.of(
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
            "Document Index",
            "Code",
            "Checkbox-Selected",
            "Checkbox-Unselected",
            "Form",
            "Key-Value Region"
    ));

    public List<List<ObjectDetectionResult>> batchPredict(List<ProcessedMat> mats, MatManager matManager, NDManager manager, float threshold) {
        try {
            int[][] targetSizes = new int[2][mats.size()];
            int i = 0;
            for (ProcessedMat mat : mats) {
                targetSizes[0][i] = mat.processed().rows();
                targetSizes[1][i] = mat.processed().cols();
                i++;
            }
            List<Mat> processed = new ArrayList<>(mats.stream().map(ProcessedMat::processed).toList());
            for (ImageProcessor processor : PREPROCESSORS) {
                processed = processor.process(matManager, processed);
            }

            List<Mat> transformedResults = processed;
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

            for (ProcessedMat mat : mats) {
                mat.release();
            }

            for (Mat mat : transformedResults) {
                mat.release();
            }

            FloatBuffer dataBuffer = FloatBuffer.wrap(floatDatas);
            long[] shape = new long[]{
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
            Optional<OnnxValue> logitsResult = onnxResult.get("logits");
            Optional<OnnxValue> predBoxesResult = onnxResult.get("pred_boxes");

            List<List<ObjectDetectionResult>> allResults = new ArrayList<>();
            if (logitsResult.isPresent() && predBoxesResult.isPresent()) {
                NDArray logits = ArrayUtil.toNDArray(manager, (float[][][]) logitsResult.get().getValue());
                NDArray bboxes = ArrayUtil.toNDArray(manager, (float[][][]) predBoxesResult.get().getValue());
                allResults.addAll(POST_PROCESSOR.process(logits, bboxes, threshold, targetSizes, true));
            }

            onnxResult.close();
            onnxInput.close();
            return allResults;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public static final float DEFAULT_THRESHOLD = 0.3f;

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return batchPredict(mats, matManager, manager, DEFAULT_THRESHOLD);
    }

    @Override
    public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

}
