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
import io.github.flux.model.LayoutModel;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Rescale;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.RtdetrPostProcessor;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class DoclingLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "docling-layout-egret-large",
            "docling-layout-egret-medium",
            "docling-layout-egret-xlarge",
            "docling-layout-heron",
            "docling-layout-heron-101"
    );

    static {
        LayoutModel.getRegistry().register(MODEL_NAMES, DoclingLayoutModel::new);
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    @Override
    public void close() throws Exception {
        session.close();
    }

    public DoclingLayoutModel(final String modelDir,
                              final String modelName,
                              final int gpuIndex,
                              final OrtEnvironment env,
                              final Map<String, Object> customParams) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            String modelFile = new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath();
            this.session = env.createSession(modelFile, options);

            this.inputName = List.copyOf(session.getInputNames()).getFirst();
        } catch (Exception e) {
            throw new FluxException(e);
        }
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

        // 修复前版本从未释放预处理最终产生的 CHW Mat：matToOnnxTensor 已将像素数据拷贝进张量，
        // 但这些 Mat 仍被长期存活的 MatManager 跟踪，导致原生内存无界累积（内存泄露）。
        // 这里把 onnxInput 移出 try-with-resources，在 finally 中保证张量与 Mat 均被释放。
        OnnxTensor onnxInput = ImageUtil.matToOnnxTensor(processed, env);
        try (OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput))) {
            Optional<OnnxValue> logitsResult = onnxResult.get("logits");
            Optional<OnnxValue> predBoxesResult = onnxResult.get("pred_boxes");

            List<List<ObjectDetectionResult>> allResults = new ArrayList<>();
            if (logitsResult.isPresent() && predBoxesResult.isPresent()) {
                // 使用独立子管理器承载 logits/bboxes 以及后处理（RtdetrPostProcessor）产生的全部
                // 临时 NDArray（约 38 个，如 scaleFct、scores、boxes 等）。这些临时资源随子管理器
                // 关闭而立即释放，不会停留在外层（可能长期存活）的 NDManager 中累积，避免 NDArray 泄露。
                try (NDManager subMgr = manager.newSubManager()) {
                    NDArray logits = ArrayUtil.toNDArray(subMgr, (float[][][]) logitsResult.get().getValue());
                    NDArray bboxes = ArrayUtil.toNDArray(subMgr, (float[][][]) predBoxesResult.get().getValue());
                    allResults.addAll(POST_PROCESSOR.process(logits, bboxes, threshold, targetSizes, true));
                }
            }

            return allResults;
        } finally {
            // 释放 ONNX 输入张量（幂等）与预处理最终产生的 CHW Mat（约 7.68MB/张，640×640×3×4B）。
            IOUtil.close(onnxInput);
            for (Mat m : processed) {
                matManager.release(m);
            }
        }
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public static final float DEFAULT_THRESHOLD = 0.3f;

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        Float threshold = ParameterUtil.getFloat(extraParameters, "docling.layout.threshold");
        return batchPredict(mats, matManager, manager, threshold==null ? DEFAULT_THRESHOLD : threshold);
    }

    @Override
    public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

}
