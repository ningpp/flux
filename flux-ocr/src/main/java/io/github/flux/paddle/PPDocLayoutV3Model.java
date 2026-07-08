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
package io.github.flux.paddle;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.exception.FluxException;
import io.github.flux.model.LayoutModel;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.PPDocLayoutV3PostProcessor;
import io.github.flux.paddle.processor.Resize;
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

/**
 * PP-DocLayoutV3 layout analysis model (DETR-style architecture with reading order).
 * <p>
 * ONNX model I/O:
 * <ul>
 *   <li>Input:  {@code pixel_values} — float32, shape (batch, 3, H, W)</li>
 *   <li>Output: {@code logits}       — float32, shape (batch, 300, 25)</li>
 *   <li>Output: {@code pred_boxes}    — float32, shape (batch, 300, 4) — normalized cxcywh</li>
 *   <li>Output: {@code order_logits}  — float32, shape (batch, 300, 300) — reading order</li>
 * </ul>
 */
public class PPDocLayoutV3Model extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    public static final Set<String> MODEL_NAMES = Set.of("PP-DocLayoutV3");

    static {
        LayoutModel.getRegistry().register(MODEL_NAMES, PPDocLayoutV3Model::new);
    }

    /** 25-class label list (identical to PP-DocLayoutV2) */
    private static final List<String> V3_LABELS = List.of(
            "abstract", "algorithm", "aside_text", "chart", "content",
            "display_formula", "doc_title", "figure_title", "footer",
            "footer_image", "footnote", "formula_number", "header",
            "header_image", "image", "inline_formula", "number",
            "paragraph_title", "reference", "reference_content", "seal",
            "table", "text", "vertical_text", "vision_footnote"
    );

    private static final List<ImageProcessor> PREPROCESSORS = List.of(
            new Resize(800, 800, Imgproc.INTER_CUBIC),
            new Normalize(1.0 / 255.0,
                    new double[]{0, 0, 0},
                    new double[]{1, 1, 1}),
            new ToCHWImage()
    );

    private static final PPDocLayoutV3PostProcessor POST_PROCESSOR =
            new PPDocLayoutV3PostProcessor(V3_LABELS);

    public static final float DEFAULT_THRESHOLD = 0.5f;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public PPDocLayoutV3Model(final String modelDir,
                               final String modelName,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               final Map<String, Object> customParams) {
        try {
            this.env = env;
            String modelFile = new File(modelDir + File.separator + modelName, "model.onnx").getAbsolutePath();
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex);
            this.inputName = List.copyOf(session.getInputNames()).getFirst();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    @Override
    public ProcessedMat processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return new ProcessedMat(rgbMat.width(), rgbMat.height(), rgbMat);
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, MatManager matManager,
                                                             NDManager manager, Map<String, Object> extraParameters) {
        try {
            // Collect original image sizes for coordinate scaling
            int[][] targetSizes = new int[2][mats.size()];
            for (int i = 0; i < mats.size(); i++) {
                targetSizes[0][i] = mats.get(i).oriHeight();
                targetSizes[1][i] = mats.get(i).oriWidth();
            }

            // Preprocess: resize → normalize → toCHW
            List<Mat> processed = new ArrayList<>(mats.stream().map(ProcessedMat::processed).toList());
            for (ImageProcessor processor : PREPROCESSORS) {
                processed = processor.process(matManager, processed);
            }

            // ONNX inference
            OnnxTensor onnxInput = ImageUtil.matToOnnxTensor(processed, env);
            try (
                    OrtSession.Result onnxResult = session.run(Map.of(inputName, onnxInput))
            ) {
                Optional<OnnxValue> logitsResult = onnxResult.get("logits");
                Optional<OnnxValue> predBoxesResult = onnxResult.get("pred_boxes");
                Optional<OnnxValue> orderLogitsResult = onnxResult.get("order_logits");

                if (logitsResult.isEmpty() || predBoxesResult.isEmpty() || orderLogitsResult.isEmpty()) {
                    throw new FluxException("PP-DocLayoutV3 model missing required outputs: logits, pred_boxes, order_logits");
                }

                NDArray logits = ArrayUtil.toNDArray(manager, (float[][][]) logitsResult.get().getValue());
                NDArray bboxes = ArrayUtil.toNDArray(manager, (float[][][]) predBoxesResult.get().getValue());
                NDArray orderLogits = ArrayUtil.toNDArray(manager, (float[][][]) orderLogitsResult.get().getValue());

                Float threshold = ParameterUtil.getFloat(extraParameters, "ppdoclayoutv3.threshold");
                float thresh = threshold != null ? threshold : DEFAULT_THRESHOLD;

                try {
                    return POST_PROCESSOR.process(logits, bboxes, orderLogits, thresh, targetSizes);
                } finally {
                    IOUtil.close(logits);
                    IOUtil.close(bboxes);
                    IOUtil.close(orderLogits);
                }
            } finally {
                // 释放 ONNX 输入张量与预处理产生的 Mat（ToCHW 输出，约 7.68MB/张），
                // 避免长期存活的 MatManager 无界累积。
                IOUtil.close(onnxInput);
                for (Mat m : processed) {
                    matManager.release(m);
                }
            }
        } catch (FluxException e) {
            throw e;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }
}
