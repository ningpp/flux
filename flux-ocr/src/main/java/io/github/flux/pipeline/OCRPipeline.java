package io.github.flux.pipeline;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.model.DocOrientationClassifyModel;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.LayoutModel;
import io.github.flux.model.TableModel;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import io.github.flux.util.IOUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

import static io.github.flux.util.ArrayUtil.splitIntoBatches;

public class OCRPipeline {

  private final BatchPredictor<PreProcessResult, TextDetectionResult> textDetectionModel;
  private final BatchPredictor<PreProcessResult, List<RecognitionResult>> textRecognitionModel;
  private final BatchPredictor<PreProcessResult, ClassificationResult> docOriClassifyModel;
  private final BatchPredictor<PreProcessResult, ClassificationResult> textLineOrientationModel;
  private final BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> layoutModel;
  private final BatchPredictor<PreProcessResult, TextResult> formulaRecognitionModel;
  private final BatchPredictor<PreProcessResult, TableResult> tableModel;

  private static final Set<String> FORMULA_LABELS = Set.of(
          "display_formula", "inline_formula",
          // Docling labels
          "Formula"
  );

  private static final Set<String> TABLE_LABELS = Set.of(
          "table",
          // Docling labels
          "Table"
  );

  private static final Set<String> IMAGE_LABELS = Set.of(
          "image", "chart", "seal", "header_image", "footer_image",
          // Docling labels
          "Picture"
  );

  public OCRPipeline(TextDetectionModel textDetectionModel, TextRecognitionModel textRecognitionModel,
                     DocOrientationClassifyModel docOriClassifyModel, TextLineOrientationModel textLineOrientationModel) {
    this(textDetectionModel, textRecognitionModel, docOriClassifyModel, textLineOrientationModel,
            null, null, null);
  }

  public OCRPipeline(TextDetectionModel textDetectionModel, TextRecognitionModel textRecognitionModel,
                     DocOrientationClassifyModel docOriClassifyModel, TextLineOrientationModel textLineOrientationModel,
                     LayoutModel layoutModel, FormulaRecognitionModel formulaRecognitionModel, TableModel tableModel) {
    this((BatchPredictor<PreProcessResult, TextDetectionResult>) textDetectionModel,
            textRecognitionModel,
            docOriClassifyModel,
            textLineOrientationModel,
            layoutModel,
            formulaRecognitionModel,
            tableModel);
  }

  OCRPipeline(BatchPredictor<PreProcessResult, TextDetectionResult> textDetectionModel,
                     BatchPredictor<PreProcessResult, List<RecognitionResult>> textRecognitionModel,
                     BatchPredictor<PreProcessResult, ClassificationResult> docOriClassifyModel,
                     BatchPredictor<PreProcessResult, ClassificationResult> textLineOrientationModel,
                     BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> layoutModel,
                     BatchPredictor<PreProcessResult, TextResult> formulaRecognitionModel,
                     BatchPredictor<PreProcessResult, TableResult> tableModel) {
    this.textDetectionModel = textDetectionModel;
    this.textRecognitionModel = textRecognitionModel;
    this.docOriClassifyModel = docOriClassifyModel;
    this.textLineOrientationModel = textLineOrientationModel;
    this.layoutModel = layoutModel;
    this.formulaRecognitionModel = formulaRecognitionModel;
    this.tableModel = tableModel;
  }

  /**
   * Batch predict: process multiple images through the OCR pipeline.
   * Returns a list of results, one per image.
   */
  public List<List<OCRPipelineResult>> predict(List<String> images, Map<String, Object> extraParameters) {
    if (images == null || images.isEmpty()) {
      return List.of();
    }

    try (
        NDManager ndManager = NDManager.newBaseManager();
        MatManager matManager = new MatManager();
    ) {
      return predict(images, extraParameters, matManager, ndManager);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  List<List<OCRPipelineResult>> predict(List<String> images, Map<String, Object> extraParameters,
                                        MatManager matManager, NDManager ndManager) {
      if (images == null || images.isEmpty()) {
        return List.of();
      }
      if (extraParameters == null) {
        extraParameters = Map.of();
      }
      markMemoryStage(extraParameters, "predict:start");

      int recognitionBatchSize = resolveBatchSize(extraParameters, "recognitionBatchSize", 1);
      int detectionBatchSize = resolveBatchSize(extraParameters, "detectionBatchSize", 1);
      int layoutBatchSize = resolveBatchSize(extraParameters, "layoutBatchSize", 1);
      int formulaBatchSize = resolveBatchSize(extraParameters, "formulaBatchSize", 1);
      int tableBatchSize = resolveBatchSize(extraParameters, "tableBatchSize", 1);

      List<Mat> bgrImages = new ArrayList<>();
      List<Mat> rgbImages = new ArrayList<>();
      List<Mat> srcImages = new ArrayList<>();
      try {
        // Read all images and convert BGR to RGB
        for (String img : images) {
          Mat bgrImage = matManager.imread(img, Imgcodecs.IMREAD_COLOR_BGR);
          Mat rgbImg = matManager.newMat();
          Imgproc.cvtColor(bgrImage, rgbImg, Imgproc.COLOR_BGR2RGB);
          bgrImages.add(bgrImage);
          rgbImages.add(rgbImg);
        }
        markMemoryStage(extraParameters, "images:loaded");

        // Batch doc orientation classification (optional)
        List<String> oriLabels = new ArrayList<>();
        List<Float> oriScores = new ArrayList<>();
        if (docOriClassifyModel != null) {
          List<PreProcessResult> oriInputs = new ArrayList<>();
          try {
            for (Mat rgbImg : rgbImages) {
              oriInputs.add(docOriClassifyModel.processRgb(matManager, rgbImg, ndManager));
            }
            List<ClassificationResult> oriResults = docOriClassifyModel.batchPredict(
                oriInputs, oriInputs.size(), matManager, ndManager, extraParameters);
            for (int i = 0; i < oriResults.size(); i++) {
              ClassificationResult oriResult = oriResults.get(i);
              oriLabels.add(oriResult.label());
              oriScores.add(oriResult.score());
            }
          } finally {
            for (PreProcessResult oriInput : oriInputs) {
              releasePreProcessResult(matManager, oriInput);
            }
          }
        } else {
          for (int i = 0; i < images.size(); i++) {
            oriLabels.add(null);
            oriScores.add(0f);
          }
        }
        markMemoryStage(extraParameters, "doc-orientation:done");
        // Release rgbImages - no longer needed after orientation classification
        for (Mat rgbImg : rgbImages) {
          matManager.release(rgbImg);
        }

        // Apply orientation correction to get srcImages
        for (int i = 0; i < images.size(); i++) {
          Mat bgrImage = bgrImages.get(i);
          String oriLabel = oriLabels.get(i);
          float oriScore = oriScores.get(i);
          Mat srcImage;
          if (oriLabel != null && oriScore > 0.3f) {
            double angle = Double.parseDouble(oriLabel);
            if (angle < 1e-7) {
              srcImage = bgrImage;
            } else {
              srcImage = ImageUtil.rotateImage(matManager, bgrImage, angle);
              matManager.release(bgrImage); // bgrImage no longer needed after rotation
            }
          } else {
            srcImage = bgrImage; // reuse bgrImage as srcImage
          }
          srcImages.add(srcImage);
        }

        List<PageContext> pages = new ArrayList<>(images.size());
        for (int i = 0; i < images.size(); i++) {
          pages.add(new PageContext(srcImages.get(i), oriLabels.get(i), oriScores.get(i)));
        }

        if (layoutModel != null) {
          List<List<ObjectDetectionResult>> allLayoutRegions =
              predictLayoutRegions(pages, layoutBatchSize, matManager, ndManager, extraParameters);
          markMemoryStage(extraParameters, "layout:done");
          List<TextTask> textTasks = new ArrayList<>();
          List<FormulaTask> formulaTasks = new ArrayList<>();
          List<TableTask> tableTasks = new ArrayList<>();
          collectLayoutTasks(pages, allLayoutRegions, textTasks, formulaTasks, tableTasks, matManager);
          processFormulaTasks(formulaTasks, formulaBatchSize, matManager, ndManager, extraParameters);
          markMemoryStage(extraParameters, "formula:done");
          processTableTasks(tableTasks, tableBatchSize, matManager, ndManager, extraParameters);
          markMemoryStage(extraParameters, "table:done");
          processTextTasks(textTasks, detectionBatchSize, recognitionBatchSize,
              matManager, ndManager, extraParameters);
          markMemoryStage(extraParameters, "text:done");

          return pages.stream()
              .map(page -> List.of(new OCRPipelineResult(
                  null, null, page.docOrientationLabel, page.docOrientationScore,
                  null, 0f, page.layoutRegionResults)))
              .toList();
        }

        List<TextTask> textTasks = new ArrayList<>();
        for (PageContext page : pages) {
          textTasks.add(new TextTask(page, null, -1, cropRegion(matManager, page.srcImage, null)));
        }
        processTextTasks(textTasks, detectionBatchSize, recognitionBatchSize,
            matManager, ndManager, extraParameters);
        markMemoryStage(extraParameters, "text:done");
        return pages.stream().map(page -> page.textResults).toList();
      } finally {
        for (Mat srcImage : srcImages) {
          matManager.release(srcImage);
        }
        for (Mat rgbImg : rgbImages) {
          matManager.release(rgbImg);
        }
        for (Mat bgrImage : bgrImages) {
          matManager.release(bgrImage);
        }
        markMemoryStage(extraParameters, "predict:released");
      }
  }

  @SuppressWarnings("unchecked")
  private void markMemoryStage(Map<String, Object> extraParameters, String stage) {
    Object observer = extraParameters.get("memoryObserver");
    if (observer instanceof Consumer<?>) {
      ((Consumer<String>) observer).accept(stage);
    }
  }

  private int resolveBatchSize(Map<String, Object> extraParameters, String key, int defaultValue) {
    Integer batchSize = ParameterUtil.getInteger(extraParameters, key);
    if (batchSize == null || batchSize < 1) {
      return Math.max(defaultValue, 1);
    }
    return batchSize;
  }

  private List<List<ObjectDetectionResult>> predictLayoutRegions(List<PageContext> pages,
                                                                 int layoutBatchSize,
                                                                 MatManager matManager,
                                                                 NDManager ndManager,
                                                                 Map<String, Object> extraParameters) {
    List<List<ObjectDetectionResult>> results = new ArrayList<>(pages.size());
    for (List<PageContext> batch : splitIntoBatches(pages, layoutBatchSize)) {
      results.addAll(predictLayoutRegionsWithRetry(batch, matManager, ndManager, extraParameters));
    }
    return results;
  }

  private List<List<ObjectDetectionResult>> predictLayoutRegionsWithRetry(List<PageContext> pages,
                                                                          MatManager matManager,
                                                                          NDManager ndManager,
                                                                          Map<String, Object> extraParameters) {
    try {
      return predictLayoutRegionsOnce(pages, matManager, ndManager, extraParameters);
    } catch (RuntimeException e) {
      if (pages.size() <= 1) {
        throw e;
      }
      int mid = pages.size() / 2;
      List<List<ObjectDetectionResult>> results = new ArrayList<>(pages.size());
      results.addAll(predictLayoutRegionsWithRetry(
          pages.subList(0, mid), matManager, ndManager, extraParameters));
      results.addAll(predictLayoutRegionsWithRetry(
          pages.subList(mid, pages.size()), matManager, ndManager, extraParameters));
      return results;
    }
  }

  private List<List<ObjectDetectionResult>> predictLayoutRegionsOnce(List<PageContext> pages,
                                                                     MatManager matManager,
                                                                     NDManager ndManager,
                                                                     Map<String, Object> extraParameters) {
    List<ProcessedMat> layoutInputs = new ArrayList<>();
    List<Mat> srcRgbs = new ArrayList<>();
    try {
      for (PageContext page : pages) {
        Mat srcRgb = matManager.newMat();
        Imgproc.cvtColor(page.srcImage, srcRgb, Imgproc.COLOR_BGR2RGB);
        srcRgbs.add(srcRgb);
        layoutInputs.add(layoutModel.processRgb(matManager, srcRgb, ndManager));
      }
      return layoutModel.batchPredict(layoutInputs, layoutInputs.size(),
          matManager, ndManager, extraParameters);
    } finally {
      for (ProcessedMat layoutInput : layoutInputs) {
        layoutInput.release(matManager);
      }
      for (Mat srcRgb : srcRgbs) {
        matManager.release(srcRgb);
      }
    }
  }

  private void collectLayoutTasks(List<PageContext> pages,
                                  List<List<ObjectDetectionResult>> allLayoutRegions,
                                  List<TextTask> textTasks,
                                  List<FormulaTask> formulaTasks,
                                  List<TableTask> tableTasks,
                                  MatManager matManager) {
    for (int pageIndex = 0; pageIndex < pages.size(); pageIndex++) {
      PageContext page = pages.get(pageIndex);
      page.layoutRegionResults = new ArrayList<>();
      List<ObjectDetectionResult> regions = pageIndex < allLayoutRegions.size()
          ? allLayoutRegions.get(pageIndex)
          : List.of();
      for (ObjectDetectionResult region : regions) {
        String regionType = classifyLabel(region.label());

        switch (regionType) {
          case "image" -> page.layoutRegionResults.add(LayoutRegionResult.image(region));
          case "formula" -> {
            if (formulaRecognitionModel != null) {
              int layoutIndex = page.layoutRegionResults.size();
              page.layoutRegionResults.add(null);
              formulaTasks.add(new FormulaTask(page, region, layoutIndex));
            } else {
              addTextTask(page, region, textTasks, matManager);
            }
          }
          case "table" -> {
            if (tableModel != null) {
              int layoutIndex = page.layoutRegionResults.size();
              page.layoutRegionResults.add(null);
              tableTasks.add(new TableTask(page, region, layoutIndex));
            } else {
              addTextTask(page, region, textTasks, matManager);
            }
          }
          default -> addTextTask(page, region, textTasks, matManager);
        }
      }
    }
  }

  private void addTextTask(PageContext page, ObjectDetectionResult region,
                           List<TextTask> textTasks, MatManager matManager) {
    int layoutIndex = page.layoutRegionResults.size();
    page.layoutRegionResults.add(null);
    textTasks.add(new TextTask(
        page, region, layoutIndex, cropRegion(matManager, page.srcImage, region.coordinate())));
  }

  /**
   * Classify a layout label into region type: "text", "formula", "table", or "image".
   */
  private String classifyLabel(String label) {
    if (FORMULA_LABELS.contains(label)) {
      return "formula";
    }
    if (TABLE_LABELS.contains(label)) {
      return "table";
    }
    if (IMAGE_LABELS.contains(label)) {
      return "image";
    }
    return "text";
  }

  /**
   * Crop a rectangular region from the source image using [x1, y1, x2, y2] coordinates.
   * If coordinate is null, returns a clone of the full source image.
   */
  private Mat cropRegion(MatManager matManager, Mat srcImage, float[] coordinate) {
    if (coordinate == null) {
      return matManager.cloneMat(srcImage);
    }
    int x1 = Math.max(0, Math.round(coordinate[0]));
    int y1 = Math.max(0, Math.round(coordinate[1]));
    int x2 = Math.min(srcImage.cols(), Math.round(coordinate[2]));
    int y2 = Math.min(srcImage.rows(), Math.round(coordinate[3]));
    if (x2 <= x1 || y2 <= y1) {
      return matManager.cloneMat(srcImage);
    }
    Rect rect = new Rect(x1, y1, x2 - x1, y2 - y1);
    return matManager.newMat(srcImage, rect);
  }

  private void processFormulaTasks(List<FormulaTask> tasks, int formulaBatchSize,
                                   MatManager matManager, NDManager ndManager,
                                   Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    for (List<FormulaTask> batch : splitIntoBatches(tasks, formulaBatchSize)) {
      processFormulaBatchWithRetry(batch, formulaBatchSize, matManager, ndManager, extraParameters);
    }
  }

  private void processFormulaBatchWithRetry(List<FormulaTask> batch,
                                            int formulaBatchSize,
                                            MatManager matManager,
                                            NDManager ndManager,
                                            Map<String, Object> extraParameters) {
    try {
      processFormulaBatchOnce(batch, formulaBatchSize, matManager, ndManager, extraParameters);
    } catch (RuntimeException e) {
      if (batch.size() <= 1) {
        throw e;
      }
      int mid = batch.size() / 2;
      processFormulaBatchWithRetry(batch.subList(0, mid), formulaBatchSize,
          matManager, ndManager, extraParameters);
      processFormulaBatchWithRetry(batch.subList(mid, batch.size()), formulaBatchSize,
          matManager, ndManager, extraParameters);
    }
  }

  private void processFormulaBatchOnce(List<FormulaTask> batch,
                                       int formulaBatchSize,
                                       MatManager matManager,
                                       NDManager ndManager,
                                       Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> rgbs = new ArrayList<>();
    try {
      for (FormulaTask task : batch) {
        Mat croppedBgr = cropRegion(matManager, task.page.srcImage, task.region.coordinate());
        Mat croppedRgb = matManager.newMat();
        Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
        matManager.release(croppedBgr);
        rgbs.add(croppedRgb);
        inputs.add(formulaRecognitionModel.processRgb(matManager, croppedRgb, ndManager));
      }
      List<TextResult> results = formulaRecognitionModel.batchPredict(
          inputs, formulaBatchSize, matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        FormulaTask task = batch.get(i);
        TextResult result = i < results.size() ? results.get(i) : null;
        task.page.layoutRegionResults.set(
            task.layoutIndex, LayoutRegionResult.formula(task.region, result));
      }
    } finally {
      for (PreProcessResult input : inputs) {
        releasePreProcessResult(matManager, input);
      }
      for (Mat rgb : rgbs) {
        matManager.release(rgb);
      }
    }
  }

  private void processTableTasks(List<TableTask> tasks, int tableBatchSize,
                                 MatManager matManager, NDManager ndManager,
                                 Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    for (List<TableTask> batch : splitIntoBatches(tasks, tableBatchSize)) {
      processTableBatchWithRetry(batch, tableBatchSize, matManager, ndManager, extraParameters);
    }
  }

  private void processTableBatchWithRetry(List<TableTask> batch,
                                          int tableBatchSize,
                                          MatManager matManager,
                                          NDManager ndManager,
                                          Map<String, Object> extraParameters) {
    try {
      processTableBatchOnce(batch, tableBatchSize, matManager, ndManager, extraParameters);
    } catch (RuntimeException e) {
      if (batch.size() <= 1) {
        throw e;
      }
      int mid = batch.size() / 2;
      processTableBatchWithRetry(batch.subList(0, mid), tableBatchSize,
          matManager, ndManager, extraParameters);
      processTableBatchWithRetry(batch.subList(mid, batch.size()), tableBatchSize,
          matManager, ndManager, extraParameters);
    }
  }

  private void processTableBatchOnce(List<TableTask> batch,
                                     int tableBatchSize,
                                     MatManager matManager,
                                     NDManager ndManager,
                                     Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> rgbs = new ArrayList<>();
    try {
      for (TableTask task : batch) {
        Mat croppedBgr = cropRegion(matManager, task.page.srcImage, task.region.coordinate());
        Mat croppedRgb = matManager.newMat();
        Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
        matManager.release(croppedBgr);
        rgbs.add(croppedRgb);
        inputs.add(tableModel.processRgb(matManager, croppedRgb, ndManager));
      }
      List<TableResult> results = tableModel.batchPredict(
          inputs, tableBatchSize, matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        TableTask task = batch.get(i);
        TableResult result = i < results.size() ? results.get(i) : null;
        task.page.layoutRegionResults.set(
            task.layoutIndex, LayoutRegionResult.table(task.region, result));
      }
    } finally {
      for (PreProcessResult input : inputs) {
        releasePreProcessResult(matManager, input);
      }
      for (Mat rgb : rgbs) {
        matManager.release(rgb);
      }
    }
  }

  private void processTextTasks(List<TextTask> tasks,
                                int detectionBatchSize,
                                int recognitionBatchSize,
                                MatManager matManager,
                                NDManager ndManager,
                                Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    try {
      for (List<TextTask> batch : splitIntoBatches(tasks, detectionBatchSize)) {
        List<TextLineTask> lineTasks = new ArrayList<>();
        try {
          lineTasks = detectTextLinesWithRetry(batch, detectionBatchSize, matManager, ndManager, extraParameters);
          markMemoryStage(extraParameters, "text-detection:done");
          for (TextTask task : batch) {
            matManager.release(task.workImage);
            task.workImage = null;
          }

          processTextLines(lineTasks, recognitionBatchSize, matManager, ndManager, extraParameters);
          for (TextTask task : batch) {
            finishTextTask(task);
          }
        } finally {
          for (TextLineTask lineTask : lineTasks) {
            matManager.release(lineTask.image);
            lineTask.image = null;
          }
        }
      }
    } finally {
      for (TextTask task : tasks) {
        matManager.release(task.workImage);
        task.workImage = null;
      }
    }
  }

  private List<TextLineTask> detectTextLinesWithRetry(List<TextTask> batch,
                                                      int detectionBatchSize,
                                                      MatManager matManager,
                                                      NDManager ndManager,
                                                      Map<String, Object> extraParameters) {
    try {
      return detectTextLinesOnce(batch, detectionBatchSize, matManager, ndManager, extraParameters);
    } catch (RuntimeException e) {
      if (batch.size() <= 1) {
        throw e;
      }
      int mid = batch.size() / 2;
      List<TextLineTask> lineTasks = new ArrayList<>();
      lineTasks.addAll(detectTextLinesWithRetry(
          batch.subList(0, mid), detectionBatchSize, matManager, ndManager, extraParameters));
      lineTasks.addAll(detectTextLinesWithRetry(
          batch.subList(mid, batch.size()), detectionBatchSize, matManager, ndManager, extraParameters));
      return lineTasks;
    }
  }

  private List<TextLineTask> detectTextLinesOnce(List<TextTask> batch,
                                                int detectionBatchSize,
                                                MatManager matManager,
                                                NDManager ndManager,
                                                Map<String, Object> extraParameters) {
    List<Mat> detectionMats = new ArrayList<>();
    List<PreProcessResult> detectionInputs = new ArrayList<>();
    List<TextLineTask> lineTasks = new ArrayList<>();
    try {
      for (TextTask task : batch) {
        Mat detectionMat = matManager.cloneMat(task.workImage);
        detectionMats.add(detectionMat);
        detectionInputs.add(new PreProcessResult(detectionMat, null));
      }
      List<TextDetectionResult> detectionResults = textDetectionModel.batchPredict(
          detectionInputs, detectionBatchSize, matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        TextDetectionResult detectionResult = i < detectionResults.size() ? detectionResults.get(i) : null;
        collectTextLineTasks(batch.get(i), detectionResult, lineTasks, matManager, ndManager);
      }
      return lineTasks;
    } finally {
      for (Mat detectionMat : detectionMats) {
        matManager.release(detectionMat);
      }
    }
  }

  private void collectTextLineTasks(TextTask task,
                                    TextDetectionResult detectionResult,
                                    List<TextLineTask> lineTasks,
                                    MatManager matManager,
                                    NDManager ndManager) {
    if (detectionResult == null || detectionResult.polys() == null || detectionResult.polys().length == 0) {
      return;
    }

    int[][][] polys = detectionResult.polys();
    NDArray polysNDArray = null;
    NDArray sortedPolysArray = null;
    try {
      polysNDArray = ArrayUtil.int3dToNDArray(ndManager, polys);
      sortedPolysArray = new SortQuadBoxes().sort(ndManager, polysNDArray);
      int[][][] sortedPolys = ArrayUtil.toInt3d(sortedPolysArray);

      for (int i = 0; i < sortedPolys.length; i++) {
        Mat lineImage = ImageUtil.getMinAreaRectCrop(matManager, ndManager, task.workImage, sortedPolys[i]);
        int[][] resultPoly = i < polys.length ? polys[i] : sortedPolys[i];
        lineTasks.add(new TextLineTask(task, resultPoly, lineImage));
      }
    } finally {
      IOUtil.close(polysNDArray);
      IOUtil.close(sortedPolysArray);
    }
  }

  private void processTextLineOrientations(List<TextLineTask> lineTasks,
                                           int batchSize,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
    if (lineTasks.isEmpty() || textLineOrientationModel == null) {
      return;
    }

    for (List<TextLineTask> batch : splitIntoBatches(lineTasks, batchSize)) {
      List<PreProcessResult> inputs = new ArrayList<>();
      List<Mat> rgbCrops = new ArrayList<>();
      try {
        for (TextLineTask lineTask : batch) {
          Mat rgbCrop = matManager.newMat();
          Imgproc.cvtColor(lineTask.image, rgbCrop, Imgproc.COLOR_BGR2RGB);
          rgbCrops.add(rgbCrop);
          inputs.add(textLineOrientationModel.processRgb(matManager, rgbCrop, ndManager));
        }

        List<ClassificationResult> orientationResults = textLineOrientationModel.batchPredict(
            inputs, batchSize, matManager, ndManager, extraParameters);
        for (int i = 0; i < batch.size(); i++) {
          if (i >= orientationResults.size()) {
            continue;
          }
          TextLineTask lineTask = batch.get(i);
          ClassificationResult orientationResult = orientationResults.get(i);
          lineTask.textLineOrientationLabel = orientationResult.label();
          lineTask.textLineOrientationScore = orientationResult.score();
          if ("180_degree".equals(orientationResult.label())) {
            Mat oldImage = lineTask.image;
            lineTask.image = ImageUtil.rotateImage(matManager, oldImage, 180.0);
            matManager.release(oldImage);
          }
        }
      } finally {
        for (PreProcessResult input : inputs) {
          releasePreProcessResult(matManager, input);
        }
        for (Mat rgbCrop : rgbCrops) {
          matManager.release(rgbCrop);
        }
      }
    }
  }

  private void processTextLines(List<TextLineTask> lineTasks,
                                int batchSize,
                                MatManager matManager,
                                NDManager ndManager,
                                Map<String, Object> extraParameters) {
    for (List<TextLineTask> batch : splitIntoBatches(lineTasks, batchSize)) {
      try {
        processTextLineOrientations(batch, batchSize, matManager, ndManager, extraParameters);
        markMemoryStage(extraParameters, "textline-orientation:done");
        processTextRecognitions(batch, batchSize, matManager, ndManager);
        markMemoryStage(extraParameters, "text-recognition:batch-done");
      } finally {
        releaseTextLineImages(batch, matManager);
      }
    }
    markMemoryStage(extraParameters, "text-recognition:done");
  }

  private void processTextRecognitions(List<TextLineTask> lineTasks,
                                       int recognitionBatchSize,
                                       MatManager matManager,
                                       NDManager ndManager) {
    if (lineTasks.isEmpty()) {
      return;
    }
    List<PreProcessResult> recognitionInputs = lineTasks.stream()
        .map(lineTask -> new PreProcessResult(lineTask.image, null))
        .toList();
    List<List<RecognitionResult>> recognitionResults = textRecognitionModel.batchPredict(
        recognitionInputs, recognitionBatchSize, matManager, ndManager, Map.of());
    for (int i = 0; i < lineTasks.size(); i++) {
      TextLineTask lineTask = lineTasks.get(i);
      List<RecognitionResult> recResults = i < recognitionResults.size()
          ? recognitionResults.get(i)
          : List.of();
      lineTask.textTask.results.add(new OCRPipelineResult(
          lineTask.detPolys,
          recResults,
          lineTask.textTask.page.docOrientationLabel,
          lineTask.textTask.page.docOrientationScore,
          lineTask.textLineOrientationLabel,
          lineTask.textLineOrientationScore));
    }
  }

  private void releaseTextLineImages(List<TextLineTask> lineTasks, MatManager matManager) {
    for (TextLineTask lineTask : lineTasks) {
      matManager.release(lineTask.image);
      lineTask.image = null;
    }
  }

  private void finishTextTask(TextTask task) {
    if (task.layoutIndex >= 0) {
      task.page.layoutRegionResults.set(
          task.layoutIndex, LayoutRegionResult.text(task.region, task.results));
    } else {
      task.page.textResults = task.results;
    }
  }

  private static final class PageContext {
    private final Mat srcImage;
    private final String docOrientationLabel;
    private final float docOrientationScore;
    private List<LayoutRegionResult> layoutRegionResults;
    private List<OCRPipelineResult> textResults = List.of();

    private PageContext(Mat srcImage, String docOrientationLabel, float docOrientationScore) {
      this.srcImage = srcImage;
      this.docOrientationLabel = docOrientationLabel;
      this.docOrientationScore = docOrientationScore;
    }
  }

  private static final class TextTask {
    private final PageContext page;
    private final ObjectDetectionResult region;
    private final int layoutIndex;
    private final List<OCRPipelineResult> results = new ArrayList<>();
    private Mat workImage;

    private TextTask(PageContext page, ObjectDetectionResult region, int layoutIndex, Mat workImage) {
      this.page = page;
      this.region = region;
      this.layoutIndex = layoutIndex;
      this.workImage = workImage;
    }
  }

  private static final class TextLineTask {
    private final TextTask textTask;
    private final int[][] detPolys;
    private Mat image;
    private String textLineOrientationLabel;
    private float textLineOrientationScore;

    private TextLineTask(TextTask textTask, int[][] detPolys, Mat image) {
      this.textTask = textTask;
      this.detPolys = detPolys;
      this.image = image;
    }
  }

  private record FormulaTask(PageContext page, ObjectDetectionResult region, int layoutIndex) {
  }

  private record TableTask(PageContext page, ObjectDetectionResult region, int layoutIndex) {
  }

  private void releasePreProcessResult(MatManager matManager, PreProcessResult ppr) {
    if (ppr == null) {
      return;
    }
    matManager.release(ppr.mat());
    IOUtil.close(ppr.ndArray());
  }

}
