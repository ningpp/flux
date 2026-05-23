package io.github.flux.falconocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.TableModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.List;

public class FalconOcrRegisteredModelDebug {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    public static void main(String[] args) throws Exception {
        String modelRootDir = args.length > 0 ? args[0] : "D:\\models";
        String modelName = args.length > 1 ? args[1] : "Falcon-OCR-ONNX";
        String imageDir = args.length > 2 ? args[2] : "D:\\models\\falcon-ocr-convert\\imgs";
        int gpuIndex = args.length > 3 ? Integer.parseInt(args[3]) : 0;

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager();
             FormulaRecognitionModel formulaModel = new FormulaRecognitionModel(modelRootDir, modelName, gpuIndex, env);
             TableModel tableModel = new TableModel(modelRootDir, modelName, gpuIndex, env)) {

            List<PreProcessResult> formulaInputs = List.of(
                    read(formulaModel, matManager, ndManager, imageDir + "\\formula-2026-01-18-152316.png"),
                    read(formulaModel, matManager, ndManager, imageDir + "\\formula_2025-8-2_17-28-16.jpg")
            );
            List<TextResult> formulaResults = formulaModel.doBatchPredict(formulaInputs, matManager, ndManager, null);
            System.out.printf("formula public entry batch=%d tokens=%d,%d%n",
                    formulaResults.size(),
                    formulaResults.get(0).tokens().length,
                    formulaResults.get(1).tokens().length);

            List<PreProcessResult> tableInputs = List.of(
                    read(tableModel, matManager, ndManager, imageDir + "\\table-2026-01-01-202211.png"),
                    read(tableModel, matManager, ndManager, imageDir + "\\table-2026-05-23-124132.png")
            );
            List<TableResult> tableResults = tableModel.doBatchPredict(tableInputs, matManager, ndManager, null);
            System.out.printf("table public entry batch=%d tokens=%d,%d%n",
                    tableResults.size(),
                    tableResults.get(0).tokens().length,
                    tableResults.get(1).tokens().length);
        }
    }

    private static PreProcessResult read(FormulaRecognitionModel model,
                                         MatManager matManager,
                                         NDManager ndManager,
                                         String path) {
        Mat rgb = ImageUtil.readToRgb(matManager, path);
        return model.processRgb(matManager, rgb, ndManager);
    }

    private static PreProcessResult read(TableModel model,
                                         MatManager matManager,
                                         NDManager ndManager,
                                         String path) {
        Mat rgb = ImageUtil.readToRgb(matManager, path);
        return model.processRgb(matManager, rgb, ndManager);
    }
}
