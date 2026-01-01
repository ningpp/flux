package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.dolphin.ByteDanceDolphinElementModel;
import io.github.flux.dolphin.ByteDanceDolphinTableModel;
import io.github.flux.exception.FluxException;
import io.github.flux.util.CollectionUtil;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class TableModel extends BatchPredictor<PreProcessResult, TableResult> {

    private final BatchPredictor<PreProcessResult, TableResult> predictor;

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            ByteDanceDolphinElementModel.MODEL_NAMES
    ));

    public TableModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env());
    }

    public TableModel(final String modelRootDir,
                      final String modelName,
                      final int gpuIndex,
                      final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported table model: " + modelName);
        }

        this.predictor = new ByteDanceDolphinTableModel(modelRootDir, modelName, gpuIndex, env, OnnxJavaType.FLOAT);
    }


    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

    @Override
    public List<TableResult> doBatchPredict(List<PreProcessResult> pprs, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(pprs, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return predictor.processRgb(rgbMat, manager);
    }

}
