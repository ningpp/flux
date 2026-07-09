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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TopkResult;
import io.github.flux.exception.FluxException;
import io.github.flux.formula.pix2text.DeiTImageProcessor;
import io.github.flux.paddle.processor.Rescale;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.ResizeNdArray;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.paddle.processor.TopkProcessor;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.util.ParameterUtil;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class DocumentFigureClassifierPredictor extends BatchPredictor<PreProcessResult, List<ClassificationResult>> {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final TopkProcessor topkProcessor;

    public DocumentFigureClassifierPredictor(final String modelDir,
                                             final int gpuIndex,
                                             final OrtEnvironment env) {
        try {
            this.env = env;
            this.session = OnnxSessionUtil.createSession(
                    env, new File(modelDir, "model.onnx").getAbsolutePath(), gpuIndex);
            this.inputName = List.copyOf(session.getInputNames()).getFirst();

            Set<Entry<String, JsonElement>> id2labelEntries = JsonParser.parseString(Files.readString(
                    new File(modelDir, "config.json").toPath(), StandardCharsets.UTF_8)
            ).getAsJsonObject().getAsJsonObject("id2label").entrySet();
            List<Pair<Integer, String>> idLables = new ArrayList<>();
            for (var entry : id2labelEntries) {
                idLables.add(Pair.of(Integer.valueOf(entry.getKey()), entry.getValue().getAsString()));
            }
            idLables.sort(Comparator.comparing(Pair::getLeft));

            this.topkProcessor = new TopkProcessor(idLables.stream().map(Pair::getRight).toList());
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<List<ClassificationResult>> doBatchPredict(List<PreProcessResult> mats, MatManager matManager,
                                                     NDManager manager, Map<String, Object> extraParameters) {
        NDList ndList = new NDList();
        NDArray inputNdArray = null;
        OnnxTensor onnxInput = null;
        OrtSession.Result onnxResult = null;
        try {
            List<List<ClassificationResult>> allResults = new ArrayList<>();
            Integer k = ParameterUtil.getInteger(extraParameters, "k");
            if (k == null) {
                k = 1;
            }

            List<NDArray> inputNDArrays = PreProcessResult.getNDArrays(mats);
            ndList.addAll(inputNDArrays);
            inputNdArray = NDArrays.stack(ndList);
            inputNDArrays.forEach(IOUtil::close);
            long[] shape = inputNdArray.getShape().getShape();
            var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            onnxResult = session.run(Map.of(inputName, onnxInput));
            OnnxValue optinalResult = onnxResult.get(0);
            float[][] preditResult = (float[][]) optinalResult.getValue();
            float[][] softmax = ArrayUtil.softmaxDim1(preditResult);
            TopkResult topkResult = topkProcessor.compute(softmax, k);
            for (int i = 0; i < mats.size(); i++) {
                List<ClassificationResult> results = new ArrayList<>();
                for (int j = 0; j < k; j++) {
                    results.add(new ClassificationResult(
                            topkResult.scores()[i][j],
                            topkResult.labels()[i][j]
                    ));
                }
                allResults.add(results);
            }
            return allResults;
        } catch (Exception e) {
            throw new FluxException(e);
        } finally {
            IOUtil.close(onnxResult);
            IOUtil.close(onnxInput);
            IOUtil.close(inputNdArray);
            IOUtil.close(ndList);
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(null, preprocess(matManager, rgbMat, ndManager));
    }

    private NDArray preprocess(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        NDArray imgNdArray = ImageUtil.toNDArrayUint8(rgbMat, ndManager);
        var resized = new ResizeNdArray(224, 224, 2).process(matManager, List.of(imgNdArray)).get(0);
        NDArray rescaled = DeiTImageProcessor.rescale(resized, 0.00392156862745098f);
        NDArray normalized = DeiTImageProcessor.normalize(rescaled,
                new float[]{0.485f, 0.456f, 0.406f}, new float[]{0.47853944f, 0.4732864f, 0.47434163f}, ndManager);
        NDArray result = normalized.transpose(2, 0, 1);
        // Release intermediate NDArrays
        IOUtil.close(imgNdArray);
        IOUtil.close(resized);
        IOUtil.close(rescaled);
        IOUtil.close(normalized);
        return result;
    }

    private Mat preprocess_wrong(MatManager matManager, Mat rgbMat) {
        Mat resized = new Resize(224, 224, 1).process(matManager, rgbMat);
        Mat rescaled = new Rescale(0.00392156862745098).process(matManager, resized);
        Mat normed = ImageUtil.normalize(matManager, rescaled,
                new Scalar(0.485, 0.456, 0.406),
                new Scalar(0.47853944, 0.4732864, 0.47434163));
        return new ToCHWImage().process(matManager, normed);
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
