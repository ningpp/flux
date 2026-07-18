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
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import io.github.flux.util.IOUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.IntStream;

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

  private static final Set<String> INLINE_FORMULA_LABELS = Set.of(
          "inline_formula"
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
      Map<String, Object> localParams = extraParameters != null
          ? new HashMap<>(extraParameters) : new HashMap<>();
      augmentMemoryObserver(localParams, matManager);
      return predict(images, localParams, matManager, ndManager);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @SuppressWarnings("unchecked")
  private void augmentMemoryObserver(Map<String, Object> params, MatManager matManager) {
    Object observer = params.get("memoryObserver");
    if (observer instanceof Consumer<?>) {
      Consumer<String> original = (Consumer<String>) observer;
      params.put("memoryObserver", (Consumer<String>) stage -> {
        original.accept(stage);
        System.out.printf("  MAT-TRACK %-30s mats=%d closeables=%d%n",
            stage, matManager.trackedMatCount(), matManager.trackedCloseableCount());
      });
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
    if (Boolean.TRUE.equals(extraParameters.get("forceGcBetweenStages"))) {
      System.gc();
      System.runFinalization();
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
    // First pass: collect table and inline formula regions per page
    Map<PageContext, List<float[]>> pageTableBoxes = new HashMap<>();
    Map<PageContext, List<float[]>> pageInlineFormulaBoxes = new HashMap<>();
    for (int pageIndex = 0; pageIndex < pages.size(); pageIndex++) {
      PageContext page = pages.get(pageIndex);
      List<ObjectDetectionResult> regions = pageIndex < allLayoutRegions.size()
          ? allLayoutRegions.get(pageIndex)
          : List.of();
      for (ObjectDetectionResult region : regions) {
        if (TABLE_LABELS.contains(region.label())) {
          pageTableBoxes.computeIfAbsent(page, k -> new ArrayList<>()).add(region.coordinate());
        }
        if (INLINE_FORMULA_LABELS.contains(region.label())) {
          pageInlineFormulaBoxes.computeIfAbsent(page, k -> new ArrayList<>()).add(region.coordinate());
        }
      }
    }

    // Second pass: classify regions and create tasks
    for (int pageIndex = 0; pageIndex < pages.size(); pageIndex++) {
      PageContext page = pages.get(pageIndex);
      page.layoutRegionResults = new ArrayList<>();
      List<ObjectDetectionResult> regions = pageIndex < allLayoutRegions.size()
          ? allLayoutRegions.get(pageIndex)
          : List.of();
      List<float[]> inlineFormulaBoxes = pageInlineFormulaBoxes.getOrDefault(page, List.of());
      for (ObjectDetectionResult region : regions) {
        String regionType = classifyLabel(region.label());

        switch (regionType) {
          case "image" -> page.layoutRegionResults.add(LayoutRegionResult.image(region));
          case "formula" -> {
            // Formula: send to formula model (if available), but skip if inside a table region
            if (formulaRecognitionModel != null
                && !isInsideAnyTable(region.coordinate(), pageTableBoxes.get(page))) {
              int layoutIndex = page.layoutRegionResults.size();
              page.layoutRegionResults.add(null);
              formulaTasks.add(new FormulaTask(page, region, layoutIndex));
            } else {
              addTextTask(page, region, textTasks, matManager, inlineFormulaBoxes);
            }
          }
          case "table" -> {
            if (tableModel != null) {
              int layoutIndex = page.layoutRegionResults.size();
              page.layoutRegionResults.add(null);
              tableTasks.add(new TableTask(page, region, layoutIndex));
            } else {
              addTextTask(page, region, textTasks, matManager, inlineFormulaBoxes);
            }
          }
          default -> addTextTask(page, region, textTasks, matManager, inlineFormulaBoxes);
        }
      }
    }
  }

  /**
   * Check if a bounding box is entirely inside any of the given table bounding boxes.
   * Used to skip formula recognition for formulas that are inside table regions
   * (the table model will handle them).
   *
   * @param formulaBox [x1, y1, x2, y2] coordinate of formula region
   * @param tableBoxes list of [x1, y1, x2, y2] table coordinates (may be null)
   * @return true if formulaBox is fully contained in any table box
   */
  private boolean isInsideAnyTable(float[] formulaBox, List<float[]> tableBoxes) {
    if (formulaBox == null || tableBoxes == null || tableBoxes.isEmpty()) {
      return false;
    }
    for (float[] tableBox : tableBoxes) {
      if (isBoxInside(formulaBox, tableBox)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check if inner box is entirely inside outer box.
   */
  private boolean isBoxInside(float[] inner, float[] outer) {
    return inner[0] >= outer[0] && inner[1] >= outer[1]
        && inner[2] <= outer[2] && inner[3] <= outer[3];
  }

  /**
   * Mask inline formula areas in a text region image with white fill.
   * Prevents text recognition from producing garbled output on formula content.
   *
   * @param image the cropped text region image (modified in place)
   * @param regionCoord [x1, y1, x2, y2] of the text region in page coordinates
   * @param inlineFormulaBoxes list of [x1, y1, x2, y2] inline formula coordinates in page space
   */
  private void maskInlineFormulas(Mat image, float[] regionCoord, List<float[]> inlineFormulaBoxes) {
    if (regionCoord == null || inlineFormulaBoxes == null || inlineFormulaBoxes.isEmpty()) {
      return;
    }
    float rx1 = regionCoord[0], ry1 = regionCoord[1], rx2 = regionCoord[2], ry2 = regionCoord[3];
    for (float[] fb : inlineFormulaBoxes) {
      // Check if formula overlaps with this text region
      float overlapX1 = Math.max(fb[0], rx1);
      float overlapY1 = Math.max(fb[1], ry1);
      float overlapX2 = Math.min(fb[2], rx2);
      float overlapY2 = Math.min(fb[3], ry2);
      if (overlapX1 < overlapX2 && overlapY1 < overlapY2) {
        // Convert to local coordinates relative to the text region
        int lx1 = Math.max(0, Math.round(overlapX1 - rx1));
        int ly1 = Math.max(0, Math.round(overlapY1 - ry1));
        int lx2 = Math.min(image.cols(), Math.round(overlapX2 - rx1));
        int ly2 = Math.min(image.rows(), Math.round(overlapY2 - ry1));
        if (lx2 > lx1 && ly2 > ly1) {
          Imgproc.rectangle(image, new Rect(lx1, ly1, lx2 - lx1, ly2 - ly1),
              new Scalar(255, 255, 255), -1);
        }
      }
    }
  }

  private void addTextTask(PageContext page, ObjectDetectionResult region,
                           List<TextTask> textTasks, MatManager matManager,
                           List<float[]> inlineFormulaBoxes) {
    int layoutIndex = page.layoutRegionResults.size();
    page.layoutRegionResults.add(null);
    Mat cropped = cropRegion(matManager, page.srcImage, region.coordinate());
    // Mask inline formula areas to prevent garbled text recognition
    if (!FORMULA_LABELS.contains(region.label())) {
      maskInlineFormulas(cropped, region.coordinate(), inlineFormulaBoxes);
    }
    textTasks.add(new TextTask(page, region, layoutIndex, cropped));
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

  // ===================== predictV2 =====================
  // 七个独立批大小 + 跨图片全局批量推理 + 流式 layout + 深拷贝裁剪（降峰值内存）
  // 绝不释放模型（线上并发，模型为共享单例）

  public List<List<OCRPipelineResult>> predictV2(List<String> images, Map<String, Object> extraParameters) {
    if (images == null || images.isEmpty()) {
      return List.of();
    }
    try (NDManager ndManager = NDManager.newBaseManager();
         MatManager matManager = new MatManager()) {
      Map<String, Object> localParams = extraParameters != null
          ? new HashMap<>(extraParameters) : new HashMap<>();
      augmentMemoryObserver(localParams, matManager);
      return predictV2(images, localParams, matManager, ndManager);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  List<List<OCRPipelineResult>> predictV2(List<String> images, Map<String, Object> extraParameters,
                                          MatManager matManager, NDManager ndManager) {
    if (images == null || images.isEmpty()) {
      return List.of();
    }
    if (extraParameters == null) {
      extraParameters = Map.of();
    }
    markMemoryStage(extraParameters, "predictV2:start");

    int layoutBatchSize = resolveBatchSize(extraParameters, "layoutBatchSize", 1);
    int docOrientationBatchSize = resolveBatchSize(extraParameters, "docOrientationBatchSize", 1);
    int textLineOrientationBatchSize = resolveBatchSize(extraParameters, "textLineOrientationBatchSize", 1);
    int detectionBatchSize = resolveBatchSize(extraParameters, "detectionBatchSize", 1);
    int recognitionBatchSize = resolveBatchSize(extraParameters, "recognitionBatchSize", 1);
    int formulaBatchSize = resolveBatchSize(extraParameters, "formulaBatchSize", 1);
    int tableBatchSize = resolveBatchSize(extraParameters, "tableBatchSize", 1);

    List<PageV2> pages = loadOrientLayoutAndCropV2(images, layoutBatchSize, docOrientationBatchSize,
        matManager, ndManager, extraParameters);

    if (layoutModel != null) {
      List<RegionTaskV2> formulaTasks = new ArrayList<>();
      List<RegionTaskV2> tableTasks = new ArrayList<>();
      List<RegionTaskV2> textTasks = new ArrayList<>();
      for (PageV2 page : pages) {
        formulaTasks.addAll(page.formulaTasks);
        tableTasks.addAll(page.tableTasks);
        textTasks.addAll(page.textTasks);
      }
      processFormulaTasksV2(formulaTasks, formulaBatchSize, matManager, ndManager, extraParameters);
      markMemoryStage(extraParameters, "formula:done");
      processTableTasksV2(tableTasks, tableBatchSize, matManager, ndManager, extraParameters);
      markMemoryStage(extraParameters, "table:done");
      processTextDetectionV2(textTasks, detectionBatchSize, matManager, ndManager, extraParameters);
      markMemoryStage(extraParameters, "text-detection:done");
      List<TextLineTaskV2> allLineTasks = new ArrayList<>();
      for (RegionTaskV2 t : textTasks) {
        allLineTasks.addAll(t.lineTasks);
      }
      processTextLineOrientationV2(allLineTasks, textLineOrientationBatchSize,
          matManager, ndManager, extraParameters);
      markMemoryStage(extraParameters, "textline-orientation:done");
      processTextRecognitionV2(allLineTasks, recognitionBatchSize, matManager, ndManager, extraParameters);
      markMemoryStage(extraParameters, "text-recognition:done");
      for (RegionTaskV2 t : textTasks) {
        finishTextTaskV2(t);
      }
      markMemoryStage(extraParameters, "text:done");
      markMemoryStage(extraParameters, "predictV2:released");
      return pages.stream()
          .map(page -> List.of(new OCRPipelineResult(
              null, null, page.docOrientationLabel, page.docOrientationScore,
              null, 0f, page.layoutRegionResults)))
          .toList();
    }

    List<RegionTaskV2> textTasks = new ArrayList<>();
    for (PageV2 page : pages) {
      textTasks.addAll(page.textTasks);
    }
    processTextDetectionV2(textTasks, detectionBatchSize, matManager, ndManager, extraParameters);
    markMemoryStage(extraParameters, "text-detection:done");
    List<TextLineTaskV2> allLineTasks = new ArrayList<>();
    for (RegionTaskV2 t : textTasks) {
      allLineTasks.addAll(t.lineTasks);
    }
    processTextLineOrientationV2(allLineTasks, textLineOrientationBatchSize,
        matManager, ndManager, extraParameters);
    markMemoryStage(extraParameters, "textline-orientation:done");
    processTextRecognitionV2(allLineTasks, recognitionBatchSize, matManager, ndManager, extraParameters);
    markMemoryStage(extraParameters, "text-recognition:done");
    for (RegionTaskV2 t : textTasks) {
      finishTextTaskV2(t);
    }
    markMemoryStage(extraParameters, "text:done");
    markMemoryStage(extraParameters, "predictV2:released");
    return pages.stream().map(page -> page.textResults).toList();
  }

  private List<PageV2> loadOrientLayoutAndCropV2(List<String> images, int layoutBatchSize,
                                                 int docOrientationBatchSize,
                                                 MatManager matManager, NDManager ndManager,
                                                 Map<String, Object> extraParameters) {
    List<PageV2> pages = new ArrayList<>(images.size());
    markMemoryStage(extraParameters, "images:loaded");
    for (List<String> subBatch : splitIntoBatches(images, layoutBatchSize)) {
      List<Mat> bgrImages = new ArrayList<>(subBatch.size());
      List<Mat> srcImages = new ArrayList<>(subBatch.size());
      List<String> oriLabels = new ArrayList<>(subBatch.size());
      List<Float> oriScores = new ArrayList<>(subBatch.size());
      try {
        for (String img : subBatch) {
          bgrImages.add(matManager.imread(img, Imgcodecs.IMREAD_COLOR_BGR));
        }
        if (docOriClassifyModel != null) {
          for (List<Integer> oriBatchIndices : splitIntoBatches(
              IntStream.range(0, bgrImages.size()).boxed().toList(), docOrientationBatchSize)) {
            List<PreProcessResult> oriInputs = new ArrayList<>();
            List<Mat> oriRgbs = new ArrayList<>();
            try {
              for (int idx : oriBatchIndices) {
                Mat rgb = matManager.newMat();
                Imgproc.cvtColor(bgrImages.get(idx), rgb, Imgproc.COLOR_BGR2RGB);
                oriRgbs.add(rgb);
                oriInputs.add(docOriClassifyModel.processRgb(matManager, rgb, ndManager));
              }
              List<ClassificationResult> oriResults = docOriClassifyModel.batchPredict(
                  oriInputs, oriInputs.size(), matManager, ndManager, extraParameters);
              for (int i = 0; i < oriBatchIndices.size(); i++) {
                int idx = oriBatchIndices.get(i);
                ClassificationResult r = oriResults.get(i);
                oriLabels.add(idx, r.label());
                oriScores.add(idx, r.score());
              }
            } finally {
              for (PreProcessResult ppr : oriInputs) releasePreProcessResult(matManager, ppr);
              for (Mat rgb : oriRgbs) matManager.release(rgb);
            }
          }
        } else {
          for (int i = 0; i < subBatch.size(); i++) {
            oriLabels.add(null);
            oriScores.add(0f);
          }
        }
        for (int i = 0; i < bgrImages.size(); i++) {
          Mat bgr = bgrImages.get(i);
          String label = oriLabels.get(i);
          float score = oriScores.get(i);
          Mat src;
          if (label != null && score > 0.3f) {
            double angle = Double.parseDouble(label);
            if (angle < 1e-7) {
              src = bgr;
              bgrImages.set(i, null);
            } else {
              src = ImageUtil.rotateImage(matManager, bgr, angle);
              matManager.release(bgr);
              bgrImages.set(i, null);
            }
          } else {
            src = bgr;
            bgrImages.set(i, null);
          }
          srcImages.add(src);
        }
        List<List<ObjectDetectionResult>> allRegions;
        if (layoutModel != null) {
          allRegions = predictLayoutWithRetryV2(srcImages, matManager, ndManager, extraParameters);
        } else {
          allRegions = Collections.nCopies(srcImages.size(), List.of());
        }
        for (int i = 0; i < srcImages.size(); i++) {
          int pageIndex = pages.size();
          PageV2 page = new PageV2(pageIndex, oriLabels.get(i), oriScores.get(i));
          pages.add(page);
          Mat src = srcImages.get(i);
          List<ObjectDetectionResult> regions = allRegions.get(i);
          if (layoutModel != null) {
            page.layoutRegionResults = new ArrayList<>();
            // First pass: collect table and inline formula regions for overlap checking
            List<float[]> tableBoxes = new ArrayList<>();
            List<float[]> inlineFormulaBoxes = new ArrayList<>();
            for (ObjectDetectionResult region : regions) {
              if (TABLE_LABELS.contains(region.label())) {
                tableBoxes.add(region.coordinate());
              }
              if (INLINE_FORMULA_LABELS.contains(region.label())) {
                inlineFormulaBoxes.add(region.coordinate());
              }
            }
            // Second pass: classify and create tasks
            for (ObjectDetectionResult region : regions) {
              String type = classifyLabel(region.label());
              switch (type) {
                case "image" -> page.layoutRegionResults.add(LayoutRegionResult.image(region));
                case "formula" -> {
                  // Formula: send to formula model (if available), but skip if inside a table region
                  if (formulaRecognitionModel != null
                      && !isInsideAnyTable(region.coordinate(), tableBoxes)) {
                    int layoutIndex = page.layoutRegionResults.size();
                    page.layoutRegionResults.add(null);
                    page.formulaTasks.add(new RegionTaskV2(page, region, layoutIndex,
                        cloneCropV2(matManager, src, region.coordinate())));
                  } else {
                    addTextRegionTaskV2(page, region, src, matManager, inlineFormulaBoxes);
                  }
                }
                case "table" -> {
                  if (tableModel != null) {
                    int layoutIndex = page.layoutRegionResults.size();
                    page.layoutRegionResults.add(null);
                    page.tableTasks.add(new RegionTaskV2(page, region, layoutIndex,
                        cloneCropV2(matManager, src, region.coordinate())));
                  } else {
                    addTextRegionTaskV2(page, region, src, matManager, inlineFormulaBoxes);
                  }
                }
                default -> addTextRegionTaskV2(page, region, src, matManager, inlineFormulaBoxes);
              }
            }
          } else {
            page.textTasks.add(new RegionTaskV2(page, null, -1, matManager.cloneMat(src)));
          }
        }
      } finally {
        for (Mat src : srcImages) matManager.release(src);
      }
    }
    markMemoryStage(extraParameters, "doc-orientation:done");
    markMemoryStage(extraParameters, "layout:done");
    return pages;
  }

  private void addTextRegionTaskV2(PageV2 page, ObjectDetectionResult region, Mat src, MatManager matManager,
                                    List<float[]> inlineFormulaBoxes) {
    int layoutIndex = page.layoutRegionResults.size();
    page.layoutRegionResults.add(null);
    Mat cropped = cloneCropV2(matManager, src, region.coordinate());
    // Mask inline formula areas to prevent garbled text recognition
    if (!FORMULA_LABELS.contains(region.label())) {
      maskInlineFormulas(cropped, region.coordinate(), inlineFormulaBoxes);
    }
    page.textTasks.add(new RegionTaskV2(page, region, layoutIndex, cropped));
  }

  private Mat cloneCropV2(MatManager matManager, Mat src, float[] coordinate) {
    if (coordinate == null) {
      return matManager.cloneMat(src);
    }
    int x1 = Math.max(0, Math.round(coordinate[0]));
    int y1 = Math.max(0, Math.round(coordinate[1]));
    int x2 = Math.min(src.cols(), Math.round(coordinate[2]));
    int y2 = Math.min(src.rows(), Math.round(coordinate[3]));
    if (x2 <= x1 || y2 <= y1) {
      return matManager.cloneMat(src);
    }
    Rect rect = new Rect(x1, y1, x2 - x1, y2 - y1);
    Mat roiMat = matManager.newMat(src, rect);
    Mat clone = matManager.cloneMat(roiMat);
    matManager.release(roiMat);
    return clone;
  }

  private List<List<ObjectDetectionResult>> predictLayoutWithRetryV2(List<Mat> srcs,
                                                                     MatManager matManager,
                                                                     NDManager ndManager,
                                                                     Map<String, Object> extraParameters) {
    return predictWithRetryV2(srcs, batch -> predictLayoutOnceV2(batch, matManager, ndManager, extraParameters));
  }

  private List<List<ObjectDetectionResult>> predictLayoutOnceV2(List<Mat> srcs,
                                                                MatManager matManager,
                                                                NDManager ndManager,
                                                                Map<String, Object> extraParameters) {
    List<ProcessedMat> inputs = new ArrayList<>();
    List<Mat> rgbs = new ArrayList<>();
    try {
      for (Mat src : srcs) {
        Mat rgb = matManager.newMat();
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_BGR2RGB);
        rgbs.add(rgb);
        inputs.add(layoutModel.processRgb(matManager, rgb, ndManager));
      }
      return layoutModel.batchPredict(inputs, inputs.size(), matManager, ndManager, extraParameters);
    } finally {
      for (ProcessedMat pm : inputs) pm.release(matManager);
      for (Mat rgb : rgbs) matManager.release(rgb);
    }
  }

  private void processFormulaTasksV2(List<RegionTaskV2> tasks, int batchSize,
                                     MatManager matManager, NDManager ndManager,
                                     Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    for (List<RegionTaskV2> batch : splitIntoBatches(tasks, batchSize)) {
      predictWithRetryV2(batch, b -> processFormulaOnceV2(b, matManager, ndManager, extraParameters));
    }
    for (RegionTaskV2 t : tasks) {
      matManager.release(t.crop);
      t.crop = null;
    }
  }

  private List<Object> processFormulaOnceV2(List<RegionTaskV2> batch,
                                            MatManager matManager, NDManager ndManager,
                                            Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> rgbs = new ArrayList<>();
    try {
      for (RegionTaskV2 t : batch) {
        Mat rgb = matManager.newMat();
        Imgproc.cvtColor(t.crop, rgb, Imgproc.COLOR_BGR2RGB);
        rgbs.add(rgb);
        inputs.add(formulaRecognitionModel.processRgb(matManager, rgb, ndManager));
      }
      List<TextResult> results = formulaRecognitionModel.batchPredict(
          inputs, inputs.size(), matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        RegionTaskV2 t = batch.get(i);
        TextResult r = i < results.size() ? results.get(i) : null;
        t.page.layoutRegionResults.set(t.layoutIndex, LayoutRegionResult.formula(t.region, r));
      }
      return List.of();
    } finally {
      for (PreProcessResult ppr : inputs) releasePreProcessResult(matManager, ppr);
      for (Mat rgb : rgbs) matManager.release(rgb);
    }
  }

  private void processTableTasksV2(List<RegionTaskV2> tasks, int batchSize,
                                   MatManager matManager, NDManager ndManager,
                                   Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    for (List<RegionTaskV2> batch : splitIntoBatches(tasks, batchSize)) {
      predictWithRetryV2(batch, b -> processTableOnceV2(b, matManager, ndManager, extraParameters));
    }
    for (RegionTaskV2 t : tasks) {
      matManager.release(t.crop);
      t.crop = null;
    }
  }

  private List<Object> processTableOnceV2(List<RegionTaskV2> batch,
                                          MatManager matManager, NDManager ndManager,
                                          Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> rgbs = new ArrayList<>();
    try {
      for (RegionTaskV2 t : batch) {
        Mat rgb = matManager.newMat();
        Imgproc.cvtColor(t.crop, rgb, Imgproc.COLOR_BGR2RGB);
        rgbs.add(rgb);
        inputs.add(tableModel.processRgb(matManager, rgb, ndManager));
      }
      List<TableResult> results = tableModel.batchPredict(
          inputs, inputs.size(), matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        RegionTaskV2 t = batch.get(i);
        TableResult r = i < results.size() ? results.get(i) : null;
        t.page.layoutRegionResults.set(t.layoutIndex, LayoutRegionResult.table(t.region, r));
      }
      return List.of();
    } finally {
      for (PreProcessResult ppr : inputs) releasePreProcessResult(matManager, ppr);
      for (Mat rgb : rgbs) matManager.release(rgb);
    }
  }

  private void processTextDetectionV2(List<RegionTaskV2> tasks, int batchSize,
                                      MatManager matManager, NDManager ndManager,
                                      Map<String, Object> extraParameters) {
    if (tasks.isEmpty()) {
      return;
    }
    for (List<RegionTaskV2> batch : splitIntoBatches(tasks, batchSize)) {
      predictWithRetryV2(batch, b -> detectTextOnceV2(b, matManager, ndManager, extraParameters));
    }
    for (RegionTaskV2 t : tasks) {
      matManager.release(t.crop);
      t.crop = null;
    }
  }

  private List<Object> detectTextOnceV2(List<RegionTaskV2> batch,
                                        MatManager matManager, NDManager ndManager,
                                        Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> detMats = new ArrayList<>();
    try {
      for (RegionTaskV2 t : batch) {
        Mat detMat = matManager.cloneMat(t.crop);
        detMats.add(detMat);
        inputs.add(new PreProcessResult(detMat, null));
      }
      List<TextDetectionResult> results = textDetectionModel.batchPredict(
          inputs, inputs.size(), matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        TextDetectionResult det = i < results.size() ? results.get(i) : null;
        collectTextLineTasksV2(batch.get(i), det, matManager, ndManager);
      }
      return List.of();
    } finally {
      for (Mat m : detMats) matManager.release(m);
    }
  }

  private void collectTextLineTasksV2(RegionTaskV2 task, TextDetectionResult detResult,
                                      MatManager matManager, NDManager ndManager) {
    if (detResult == null || detResult.polys() == null || detResult.polys().length == 0) {
      return;
    }
    int[][][] polys = detResult.polys();
    NDArray polysNDArray = null;
    NDArray sortedPolysArray = null;
    try {
      polysNDArray = ArrayUtil.int3dToNDArray(ndManager, polys);
      sortedPolysArray = new SortQuadBoxes().sort(ndManager, polysNDArray);
      int[][][] sortedPolys = ArrayUtil.toInt3d(sortedPolysArray);
      for (int i = 0; i < sortedPolys.length; i++) {
        Mat lineImage = ImageUtil.getMinAreaRectCrop(matManager, ndManager, task.crop, sortedPolys[i]);
        int[][] resultPoly = i < polys.length ? polys[i] : sortedPolys[i];
        task.lineTasks.add(new TextLineTaskV2(task, resultPoly, lineImage));
      }
    } finally {
      IOUtil.close(polysNDArray);
      IOUtil.close(sortedPolysArray);
    }
  }

  private void processTextLineOrientationV2(List<TextLineTaskV2> lineTasks, int batchSize,
                                            MatManager matManager, NDManager ndManager,
                                            Map<String, Object> extraParameters) {
    if (lineTasks.isEmpty() || textLineOrientationModel == null) {
      return;
    }
    for (List<TextLineTaskV2> batch : splitIntoBatches(lineTasks, batchSize)) {
      List<PreProcessResult> inputs = new ArrayList<>();
      List<Mat> rgbs = new ArrayList<>();
      try {
        for (TextLineTaskV2 lt : batch) {
          Mat rgb = matManager.newMat();
          Imgproc.cvtColor(lt.image, rgb, Imgproc.COLOR_BGR2RGB);
          rgbs.add(rgb);
          inputs.add(textLineOrientationModel.processRgb(matManager, rgb, ndManager));
        }
        List<ClassificationResult> results = textLineOrientationModel.batchPredict(
            inputs, inputs.size(), matManager, ndManager, extraParameters);
        for (int i = 0; i < batch.size(); i++) {
          if (i >= results.size()) {
            continue;
          }
          TextLineTaskV2 lt = batch.get(i);
          ClassificationResult r = results.get(i);
          lt.textLineOrientationLabel = r.label();
          lt.textLineOrientationScore = r.score();
          if ("180_degree".equals(r.label())) {
            Mat old = lt.image;
            lt.image = ImageUtil.rotateImage(matManager, old, 180.0);
            matManager.release(old);
          }
        }
      } finally {
        for (PreProcessResult ppr : inputs) releasePreProcessResult(matManager, ppr);
        for (Mat rgb : rgbs) matManager.release(rgb);
      }
    }
  }

  private void processTextRecognitionV2(List<TextLineTaskV2> lineTasks, int batchSize,
                                        MatManager matManager, NDManager ndManager,
                                        Map<String, Object> extraParameters) {
    if (lineTasks.isEmpty()) {
      return;
    }
    for (List<TextLineTaskV2> batch : splitIntoBatches(lineTasks, batchSize)) {
      predictWithRetryV2(batch, b -> recognizeTextOnceV2(b, matManager, ndManager, extraParameters));
    }
    for (TextLineTaskV2 lt : lineTasks) {
      matManager.release(lt.image);
      lt.image = null;
    }
  }

  private List<Object> recognizeTextOnceV2(List<TextLineTaskV2> batch,
                                           MatManager matManager, NDManager ndManager,
                                           Map<String, Object> extraParameters) {
    List<PreProcessResult> inputs = new ArrayList<>();
    List<Mat> recMats = new ArrayList<>();
    try {
      for (TextLineTaskV2 lt : batch) {
        Mat recMat = matManager.cloneMat(lt.image);
        recMats.add(recMat);
        inputs.add(new PreProcessResult(recMat, null));
      }
      List<List<RecognitionResult>> results = textRecognitionModel.batchPredict(
          inputs, inputs.size(), matManager, ndManager, extraParameters);
      for (int i = 0; i < batch.size(); i++) {
        TextLineTaskV2 lt = batch.get(i);
        List<RecognitionResult> rec = i < results.size() ? results.get(i) : List.of();
        lt.textTask.results.add(new OCRPipelineResult(
            lt.detPolys, rec,
            lt.textTask.page.docOrientationLabel, lt.textTask.page.docOrientationScore,
            lt.textLineOrientationLabel, lt.textLineOrientationScore));
      }
      return List.of();
    } finally {
      for (Mat m : recMats) matManager.release(m);
    }
  }

  private void finishTextTaskV2(RegionTaskV2 task) {
    if (task.layoutIndex >= 0) {
      task.page.layoutRegionResults.set(
          task.layoutIndex, LayoutRegionResult.text(task.region, task.results));
    } else {
      task.page.textResults = task.results;
    }
  }

  @FunctionalInterface
  private interface V2BatchFunction<T, R> {
    List<R> apply(List<T> batch) throws RuntimeException;
  }

  private <T, R> List<R> predictWithRetryV2(List<T> batch, V2BatchFunction<T, R> function) {
    try {
      return function.apply(batch);
    } catch (RuntimeException e) {
      if (batch.size() <= 1) {
        throw e;
      }
      int mid = batch.size() / 2;
      List<R> out = new ArrayList<>(batch.size());
      out.addAll(predictWithRetryV2(batch.subList(0, mid), function));
      out.addAll(predictWithRetryV2(batch.subList(mid, batch.size()), function));
      return out;
    }
  }

  private static final class PageV2 {
    final int pageIndex;
    final String docOrientationLabel;
    final float docOrientationScore;
    List<LayoutRegionResult> layoutRegionResults = List.of();
    List<OCRPipelineResult> textResults = List.of();
    final List<RegionTaskV2> textTasks = new ArrayList<>();
    final List<RegionTaskV2> formulaTasks = new ArrayList<>();
    final List<RegionTaskV2> tableTasks = new ArrayList<>();

    PageV2(int pageIndex, String docOrientationLabel, float docOrientationScore) {
      this.pageIndex = pageIndex;
      this.docOrientationLabel = docOrientationLabel;
      this.docOrientationScore = docOrientationScore;
    }
  }

  private static final class RegionTaskV2 {
    final PageV2 page;
    final ObjectDetectionResult region;
    final int layoutIndex;
    final List<OCRPipelineResult> results = new ArrayList<>();
    final List<TextLineTaskV2> lineTasks = new ArrayList<>();
    Mat crop;

    RegionTaskV2(PageV2 page, ObjectDetectionResult region, int layoutIndex, Mat crop) {
      this.page = page;
      this.region = region;
      this.layoutIndex = layoutIndex;
      this.crop = crop;
    }
  }

  private static final class TextLineTaskV2 {
    final RegionTaskV2 textTask;
    final int[][] detPolys;
    Mat image;
    String textLineOrientationLabel;
    float textLineOrientationScore;

    TextLineTaskV2(RegionTaskV2 textTask, int[][] detPolys, Mat image) {
      this.textTask = textTask;
      this.detPolys = detPolys;
      this.image = image;
    }
  }

}
