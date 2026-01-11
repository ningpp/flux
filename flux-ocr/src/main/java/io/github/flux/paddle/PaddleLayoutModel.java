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

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.exception.FluxException;
import io.github.flux.paddle.predictor.PaddleObjectDetectionPredictor;
import io.github.flux.paddle.processor.DetPostProcessor;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.ObjectDetectionResize;
import io.github.flux.paddle.processor.ResizeForObjectDect;
import io.github.flux.paddle.processor.ToCHWImage;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.Map;

public class PaddleLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {

    public static final List<String> MODEL_NAMES = List.of(
            "PP-DocLayoutV2",
            "PP-DocLayout_plus-L",
            "PP-DocLayout-L",
            "PP-DocLayout-M",
            "PP-DocLayout-S",
            "PicoDet-S_layout_17cls",
            "PicoDet-L_layout_17cls",
            "RT-DETR-H_layout_17cls"
    );

    private static final List<String> PP_LABELS_17 = List.of(
            "paragraph_title", "image", "text", "number", "abstract", "content",
            "figure_title", "formula", "table", "table_title", "reference",
            "doc_title", "footnote", "header", "algorithm", "footer", "seal"
    );

    private static final List<String> PP_LABELS_20 = List.of(
            "paragraph_title", "image", "text", "number", "abstract", "content",
            "figure_title", "formula", "table", "reference", "doc_title",
            "footnote", "header", "algorithm", "footer", "seal", "chart",
            "formula_number", "aside_text", "reference_content"
    );

    private static final List<String> PP_LABELS_23 = List.of(
            "paragraph_title",
            "image",
            "text",
            "number",
            "abstract",
            "content",
            "figure_title",
            "formula",
            "table",
            "table_title",
            "reference",
            "doc_title",
            "footnote",
            "header",
            "algorithm",
            "footer",
            "seal",
            "chart_title",
            "chart",
            "formula_number",
            "header_image",
            "footer_image",
            "aside_text"
    );

    private static final List<String> V2_LABELS = List.of(
            "abstract",
            "algorithm",
            "aside_text",
            "chart",
            "content",
            "display_formula",
            "doc_title",
            "figure_title",
            "footer",
            "footer_image",
            "footnote",
            "formula_number",
            "header",
            "header_image",
            "image",
            "inline_formula",
            "number",
            "paragraph_title",
            "reference",
            "reference_content",
            "seal",
            "table",
            "text",
            "vertical_text",
            "vision_footnote"
    );

    private final PaddleObjectDetectionPredictor predictor;

    public PaddleLayoutModel(final String modelDir,
                             final String modelName,
                             final int gpuIndex,
                             final OrtEnvironment env) {
        int resizeSize;
        Normalize normalize;
        DetPostProcessor detPostProcessor;
        if ("PP-DocLayout-L".equals(modelName)) {
            resizeSize = 640;
            normalize = new Normalize(1.0 / 255.0, new double[]{0, 0, 0}, new double[]{1, 1, 1});
            detPostProcessor = new DetPostProcessor(PP_LABELS_23);
        } else if ("PP-DocLayout-M".equals(modelName)) {
            resizeSize = 640;
            normalize = new Normalize(1.0 / 255.0, new double[]{0.485, 0.456, 0.406}, new double[]{0.229, 0.224, 0.225});
            detPostProcessor = new DetPostProcessor(PP_LABELS_23);
        } else if ("PP-DocLayout-S".equals(modelName)) {
            resizeSize = 480;
            normalize = new Normalize(1.0 / 255.0, new double[]{0.485, 0.456, 0.406}, new double[]{0.229, 0.224, 0.225});
            detPostProcessor = new DetPostProcessor(PP_LABELS_23);
        } else if ("PP-DocLayout_plus-L".equals(modelName)) {
            resizeSize = 800;
            normalize = new Normalize(1.0 / 255.0, new double[]{0, 0, 0}, new double[]{1, 1, 1});
            detPostProcessor = new DetPostProcessor(PP_LABELS_20);
        } else if ("PicoDet-S_layout_17cls".equals(modelName)) {
            resizeSize = 480;
            normalize = new Normalize(1.0 / 255.0, new double[]{0.485, 0.456, 0.406}, new double[]{0.229, 0.224, 0.225});
            detPostProcessor = new DetPostProcessor(PP_LABELS_17);
        } else if ("PicoDet-L_layout_17cls".equals(modelName)) {
            resizeSize = 640;
            normalize = new Normalize(1.0 / 255.0, new double[]{0.485, 0.456, 0.406}, new double[]{0.229, 0.224, 0.225});
            detPostProcessor = new DetPostProcessor(PP_LABELS_17);
        } else if ("RT-DETR-H_layout_17cls".equals(modelName)) {
            resizeSize = 640;
            normalize = new Normalize(1.0 / 255.0, new double[]{0, 0, 0}, new double[]{1, 1, 1});
            detPostProcessor = new DetPostProcessor(PP_LABELS_17);
        } else if ("PP-DocLayoutV2".equals(modelName)) {
            resizeSize = 800;
            normalize = new Normalize(1.0 / 255.0, new double[]{0, 0, 0}, new double[]{1, 1, 1});
            detPostProcessor = new DetPostProcessor(V2_LABELS);
        } else {
            throw new FluxException("Not Support Model: " + modelName);
        }
        List<ImageProcessor> preProcessors = List.of(
                normalize,
                new ToCHWImage()
        );
        ObjectDetectionResize detResize = new ObjectDetectionResize(
                new ResizeForObjectDect(resizeSize, resizeSize, Imgproc.INTER_CUBIC)
        );

        this.predictor = new PaddleObjectDetectionPredictor(modelDir, modelName, gpuIndex, env,
                0.5f, detResize, preProcessors, detPostProcessor);
    }

    @Override
    public List<List<ObjectDetectionResult>> doBatchPredict(List<ProcessedMat> mats, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.predict(mats, manager, true);
    }

    @Override
    public ProcessedMat processRgb(Mat rgbMat, NDManager manager) {
        return predictor.processRgb(rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }
}
