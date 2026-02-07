package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

public class UnirecTableModel extends BatchPredictor<PreProcessResult, TableResult> {

    private final UnirecPredictor predictor;

    public UnirecTableModel(UnirecPredictor predictor) {
        this.predictor = predictor;
    }

    public UnirecTableModel(final String modelRootDir,
                            final String modelName,
                            final int gpuIndex,
                            final OrtEnvironment env,
                            final Map<String, Object> customParams) {
        this.predictor = UnirecPredictor.getSharedInstance(modelRootDir, modelName, gpuIndex, env, customParams);
    }

    @Override
    public List<TableResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        return predictor.doBatchPredict(mats, matManager, manager, extraParameters).stream()
                .map(r -> new TableResult(r.text(), r.tokens())).toList();
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager manager) {
        return predictor.processRgb(matManager, rgbMat, manager);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(predictor);
    }
}
