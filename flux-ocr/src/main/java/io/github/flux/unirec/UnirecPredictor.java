// this code is convert from https://github.com/Topdu/OpenOCR
// OpenOCR's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.PreProcessResult;
import io.github.flux.dolphin.DolphinElementResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class UnirecPredictor extends BatchPredictor<PreProcessResult, DolphinElementResult> {

    public static final List<String> MODEL_NAMES = List.of(
            "unirec-0.1b"
    );

    private final UnirecEncoderModel encoderModel;
    private final UnirecDecoderModel decoderModel;

    public UnirecPredictor(final String modelRootDir,
                           final String modelName,
                           final int gpuIndex,
                           final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported unirec model: " + modelName);
        }
        this.encoderModel = new UnirecEncoderModel(
                new File(modelRootDir + File.separator + modelName, "unirec_encoder.onnx").getAbsolutePath(), gpuIndex, env);
        this.decoderModel = new UnirecDecoderModel(
                new File(modelRootDir + File.separator + modelName, "unirec_decoder.onnx").getAbsolutePath(),
                new File(modelRootDir + File.separator + modelName, "unirec_tokenizer_mapping.json").getAbsolutePath(),
                gpuIndex, env);
    }

    @Override
    public List<DolphinElementResult> doBatchPredict(List<PreProcessResult> mats, NDManager manager, Map<String, Object> extraParameters) {
        List<DolphinElementResult> results = new ArrayList<>(mats.size());
        for (PreProcessResult ppr : mats) {
            UnirecEncoderModelPredictResult encodeResult = encoderModel.predict(ppr, manager);
            results.add(decoderModel.predict(encodeResult));
        }
        return results;
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return new PreProcessResult(new UnirecProcessor().process(rgbMat), null);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
    }
}
