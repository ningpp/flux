package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelParam;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.dolphin.ByteDanceDolphinElementModel;
import io.github.flux.dolphin.ByteDanceDolphinTableModel;
import io.github.flux.exception.FluxException;
import io.github.flux.unirec.UnirecPredictor;
import io.github.flux.unirec.UnirecTableModel;
import io.github.flux.util.CollectionUtil;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class TableModel extends BatchPredictor<PreProcessResult, TableResult> {

    private final BatchPredictor<PreProcessResult, TableResult> predictor;

    public static final Set<String> MODEL_NAMES = CollectionUtil.distinct(List.of(
            ByteDanceDolphinElementModel.MODEL_NAMES,
            UnirecPredictor.MODEL_NAMES
    ));

    public TableModel(ModelParam param) {
        this(param.modelRootDir(), param.modelName(), param.gpuIndex(), param.env());
    }

    public TableModel(final String modelRootDir,
                      final String modelName,
                      final int gpuIndex,
                      final OrtEnvironment env) {
        if (ByteDanceDolphinElementModel.MODEL_NAMES.contains(modelName)) {
            this.predictor = new ByteDanceDolphinTableModel(modelRootDir, modelName, gpuIndex, env);
        } else if (UnirecPredictor.MODEL_NAMES.contains(modelName)) {
            this.predictor = new UnirecTableModel(new UnirecPredictor(modelRootDir, modelName, gpuIndex, env));
        } else {
            throw new FluxException("not supported table model: " + modelName);
        }
    }


    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }

    @Override
    public List<TableResult> doBatchPredict(List<PreProcessResult> pprs, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(pprs, matManager, manager, extraParameters);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

}
