package io.github.flux.falconocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class FalconOcrTableModel extends BatchPredictor<PreProcessResult, TableResult> {

    private final FalconOcrModel model;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public FalconOcrTableModel(final String modelRootDir,
                               final String modelName,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               final Map<String, Object> customParams) {
        this.model = FalconOcrModel.getSharedInstance(modelRootDir, modelName, gpuIndex, env, customParams);
    }

    @Override
    public List<TableResult> doBatchPredict(List<PreProcessResult> images,
                                            MatManager matManager,
                                            NDManager ndManager,
                                            Map<String, Object> extraParameters) {
        List<TextResult> results = model.predictCategory(images, matManager, "table");
        return results.stream().map(r -> new TableResult(r.text(), r.tokens())).toList();
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return model.processRgb(matManager, rgbMat, ndManager);
    }

    @Override
    public void close() throws Exception {
        if (closed.compareAndSet(false, true)) {
            model.close();
        }
    }
}
