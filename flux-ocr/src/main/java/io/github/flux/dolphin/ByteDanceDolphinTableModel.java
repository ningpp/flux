package io.github.flux.dolphin;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class ByteDanceDolphinTableModel extends BatchPredictor<PreProcessResult, TableResult> {

    private final ByteDanceDolphinElementModel model;

    public ByteDanceDolphinTableModel(ByteDanceDolphinElementModel model) {
        this.model = model;
    }

    public ByteDanceDolphinTableModel(final String modelRootDir,
                                      final String modelName,
                                      final int gpuIndex,
                                      final OrtEnvironment env,
                                      final OnnxJavaType dtype) {
        this.model = new ByteDanceDolphinElementModel(modelRootDir, modelName, gpuIndex, env, dtype, false);
    }

    @Override
    public List<TableResult> doBatchPredict(List<PreProcessResult> mats, NDManager manager, Map<String, Object> extraParameters) {
        extraParameters.put("prompt", "Parse the table in the image.");
        var elementResults = model.doBatchPredict(mats, manager, extraParameters);
        return elementResults.stream().map(er -> new TableResult(er.text(), er.tokens())).toList();
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return model.processRgb(rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(model);
    }
}
