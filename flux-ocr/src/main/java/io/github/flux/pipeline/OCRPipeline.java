package io.github.flux.pipeline;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.github.flux.util.ArrayUtil.splitIntoBatches;

public class OCRPipeline {

  private final TextDetectionModel textDetectionModel;
  private final TextRecognitionModel textRecognitionModel;
  private final DocOrientationClassifyModel docOriClassifyModel;
  private final TextLineOrientationModel textLineOrientationModel;
  private final LayoutModel layoutModel;
  private final FormulaRecognitionModel formulaRecognitionModel;
  private final TableModel tableModel;

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
      Integer recognitionBatchSize = ParameterUtil.getInteger(extraParameters, "recognitionBatchSize");
      if (recognitionBatchSize == null || recognitionBatchSize < 1) {
        recognitionBatchSize = 1;
      }

      Integer formulaBatchSize = ParameterUtil.getInteger(extraParameters, "formulaBatchSize");
      if (formulaBatchSize == null || formulaBatchSize < 1) {
        formulaBatchSize = 1;
      }

      // Read all images and convert BGR to RGB
      List<Mat> bgrImages = new ArrayList<>();
      List<Mat> rgbImages = new ArrayList<>();
      for (String img : images) {
        Mat bgrImage = matManager.imread(img, Imgcodecs.IMREAD_COLOR_BGR);
        Mat rgbImg = matManager.newMat();
        Imgproc.cvtColor(bgrImage, rgbImg, Imgproc.COLOR_BGR2RGB);
        bgrImages.add(bgrImage);
        rgbImages.add(rgbImg);
      }

      // Batch doc orientation classification (optional)
      List<String> oriLabels = new ArrayList<>();
      List<Float> oriScores = new ArrayList<>();
      if (docOriClassifyModel != null) {
        List<PreProcessResult> oriInputs = new ArrayList<>();
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
      } else {
        for (int i = 0; i < images.size(); i++) {
          oriLabels.add(null);
          oriScores.add(0f);
        }
      }

      // Apply orientation correction to get srcImages
      List<Mat> srcImages = new ArrayList<>();
      for (int i = 0; i < images.size(); i++) {
        Mat bgrImage = bgrImages.get(i);
        String oriLabel = oriLabels.get(i);
        float oriScore = oriScores.get(i);
        Mat srcImage;
        if (oriLabel != null && oriScore > 0.3f) {
          srcImage = ImageUtil.rotateImage(matManager, bgrImage, Double.parseDouble(oriLabel));
        } else {
          srcImage = bgrImage;
        }
        srcImages.add(srcImage);
      }

      // Batch layout analysis (optional)
      List<List<ObjectDetectionResult>> allLayoutRegions = new ArrayList<>();
      if (layoutModel != null) {
        List<ProcessedMat> layoutInputs = new ArrayList<>();
        List<Mat> srcRgbs = new ArrayList<>();
        for (Mat srcImage : srcImages) {
          Mat srcRgb = matManager.newMat();
          Imgproc.cvtColor(srcImage, srcRgb, Imgproc.COLOR_BGR2RGB);
          srcRgbs.add(srcRgb);
          layoutInputs.add(layoutModel.processRgb(matManager, srcRgb, ndManager));
        }
        List<List<ObjectDetectionResult>> layoutResults = layoutModel.batchPredict(
            layoutInputs, layoutInputs.size(), matManager, ndManager, extraParameters);
        for (Mat srcRgb : srcRgbs) {
          srcRgb.release();
        }
        for (int i = 0; i < layoutResults.size(); i++) {
          allLayoutRegions.add(layoutResults.get(i));
        }
      }

      // Process each image
      List<List<OCRPipelineResult>> allResults = new ArrayList<>();
      for (int i = 0; i < images.size(); i++) {
        Mat srcImage = srcImages.get(i);
        String oriLabel = oriLabels.get(i);
        float oriScore = oriScores.get(i);

        List<OCRPipelineResult> imageResults;
        if (layoutModel != null) {
          List<ObjectDetectionResult> regions = allLayoutRegions.get(i);
          imageResults = processLayoutRegions(matManager, ndManager, srcImage, regions,
              oriLabel, oriScore, recognitionBatchSize, formulaBatchSize, extraParameters);
        } else {
          imageResults = recognizeText(matManager, ndManager, srcImage, null,
              oriLabel, oriScore, recognitionBatchSize, extraParameters);
        }
        allResults.add(imageResults);
      }

      // Release resources
      for (int i = 0; i < images.size(); i++) {
        srcImages.get(i).release();
        bgrImages.get(i).release();
        if (oriLabels.get(i) == null || oriScores.get(i) <= 0.3f) {
          rgbImages.get(i).release();
        }
      }

      return allResults;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Process layout regions for a single image.
   */
  private List<OCRPipelineResult> processLayoutRegions(MatManager matManager, NDManager ndManager,
                                                        Mat srcImage, List<ObjectDetectionResult> regions,
                                                        String oriLabel, float oriScore,
                                                        int recognitionBatchSize, int formulaBatchSize,
                                                        Map<String, Object> extraParameters) {
    List<LayoutRegionResult> layoutRegionResults = new ArrayList<>();

    // Collect formula regions for batch processing
    List<Integer> formulaRegionIndices = new ArrayList<>();
    List<ObjectDetectionResult> formulaRegions = new ArrayList<>();
    List<PreProcessResult> formulaInputs = new ArrayList<>();
    List<Mat> formulaRgbs = new ArrayList<>();

    for (int i = 0; i < regions.size(); i++) {
      ObjectDetectionResult region = regions.get(i);
      String regionType = classifyLabel(region.label());

      switch (regionType) {
        case "image" -> {
          layoutRegionResults.add(LayoutRegionResult.image(region));
        }
        case "formula" -> {
          if (formulaRecognitionModel != null) {
            Mat croppedBgr = cropRegion(matManager, srcImage, region.coordinate());
            Mat croppedRgb = matManager.newMat();
            Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
            formulaRgbs.add(croppedRgb);
            PreProcessResult formulaInput = formulaRecognitionModel.processRgb(matManager, croppedRgb, ndManager);
            formulaInputs.add(formulaInput);
            formulaRegions.add(region);
            formulaRegionIndices.add(layoutRegionResults.size());
            layoutRegionResults.add(null); // placeholder
          } else {
            List<OCRPipelineResult> textResults = recognizeText(
                matManager, ndManager, srcImage, region.coordinate(),
                oriLabel, oriScore, recognitionBatchSize, extraParameters);
            layoutRegionResults.add(LayoutRegionResult.text(region, textResults));
          }
        }
        case "table" -> {
          if (tableModel != null) {
            Mat croppedBgr = cropRegion(matManager, srcImage, region.coordinate());
            Mat croppedRgb = matManager.newMat();
            Imgproc.cvtColor(croppedBgr, croppedRgb, Imgproc.COLOR_BGR2RGB);
            PreProcessResult tableInput = tableModel.processRgb(matManager, croppedRgb, ndManager);
            List<TableResult> tableResults = tableModel.batchPredict(
                List.of(tableInput), 1, matManager, ndManager, extraParameters);
            croppedRgb.release();
            TableResult tableResult = tableResults.isEmpty() ? null : tableResults.get(0);
            layoutRegionResults.add(LayoutRegionResult.table(region, tableResult));
          } else {
            List<OCRPipelineResult> textResults = recognizeText(
                matManager, ndManager, srcImage, region.coordinate(),
                oriLabel, oriScore, recognitionBatchSize, extraParameters);
            layoutRegionResults.add(LayoutRegionResult.text(region, textResults));
          }
        }
        default -> {
          List<OCRPipelineResult> textResults = recognizeText(
              matManager, ndManager, srcImage, region.coordinate(),
              oriLabel, oriScore, recognitionBatchSize, extraParameters);
          layoutRegionResults.add(LayoutRegionResult.text(region, textResults));
        }
      }
    }

    // Batch process formula regions
    if (!formulaInputs.isEmpty()) {
      List<List<PreProcessResult>> batched = splitIntoBatches(formulaInputs, formulaBatchSize);
      List<TextResult> allFormulaResults = new ArrayList<>();
      for (List<PreProcessResult> batch : batched) {
        List<TextResult> batchResults = formulaRecognitionModel.batchPredict(
            batch, batch.size(), matManager, ndManager, extraParameters);
        allFormulaResults.addAll(batchResults);
      }
      // Fill placeholders with formula results
      for (int i = 0; i < formulaRegionIndices.size(); i++) {
        int layoutIndex = formulaRegionIndices.get(i);
        ObjectDetectionResult region = formulaRegions.get(i);
        TextResult formulaResult = (i < allFormulaResults.size()) ? allFormulaResults.get(i) : null;
        layoutRegionResults.set(layoutIndex, LayoutRegionResult.formula(region, formulaResult));
      }
      // Release formula RGB mats
      for (Mat rgb : formulaRgbs) {
        rgb.release();
      }
    }

    return List.of(new OCRPipelineResult(
        null, null, oriLabel, oriScore, null, 0f, layoutRegionResults));
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

  /**
   * Run text detection + recognition on an image (full or cropped region).
   *
   * @param coordinate optional [x1, y1, x2, y2] to crop a region; null for full image
   * @param docOriLabel document orientation label (passed through to results)
   * @param docOriScore document orientation score (passed through to results)
   */
  private List<OCRPipelineResult> recognizeText(MatManager matManager, NDManager ndManager,
                                                 Mat srcImage, float[] coordinate,
                                                 String docOriLabel, float docOriScore,
                                                 int recognitionBatchSize,
                                                 Map<String, Object> extraParameters) {
    List<OCRPipelineResult> results = new ArrayList<>();
    Mat workImage = cropRegion(matManager, srcImage, coordinate);

    var detectionResult = textDetectionModel.batchPredict(
        List.of(new PreProcessResult(matManager.cloneMat(workImage), null)), 1,
        matManager, ndManager, extraParameters).get(0);

    int[][][] polys = detectionResult.polys();
    if (polys == null || polys.length == 0) {
      return results;
    }

    NDArray polysNDArray = ArrayUtil.int3dToNDArray(ndManager, polys);
    NDArray sortedPolysArray = new SortQuadBoxes().sort(ndManager, polysNDArray);
    int[][][] sortedPolys = ArrayUtil.toInt3d(sortedPolysArray);
    List<Mat> cropedImages = new ArrayList<>();

    for (int[][] poly : sortedPolys) {
      cropedImages.add(ImageUtil.getMinAreaRectCrop(matManager, ndManager, workImage, poly));
    }

    // Text line orientation classification: if 180_degree, rotate the cropped image
    List<String> textLineOriLabels = new ArrayList<>();
    List<Float> textLineOriScores = new ArrayList<>();
    if (textLineOrientationModel != null) {
      List<PreProcessResult> textLineInputs = new ArrayList<>();
      for (Mat cropedImg : cropedImages) {
        Mat rgbCrop = matManager.newMat();
        Imgproc.cvtColor(cropedImg, rgbCrop, Imgproc.COLOR_BGR2RGB);
        textLineInputs.add(textLineOrientationModel.processRgb(matManager, rgbCrop, ndManager));
      }
      List<ClassificationResult> textLineOriResults = textLineOrientationModel.batchPredict(
          textLineInputs, textLineInputs.size(), matManager, ndManager, extraParameters);
      for (int i = 0; i < textLineOriResults.size(); i++) {
        ClassificationResult oriResult = textLineOriResults.get(i);
        textLineOriLabels.add(oriResult.label());
        textLineOriScores.add(oriResult.score());
        if ("180_degree".equals(oriResult.label())) {
          Mat rotated = ImageUtil.rotateImage(matManager, cropedImages.get(i), 180.0);
          cropedImages.set(i, rotated);
        }
      }
    }

    int index = 0;
    List<List<Mat>> batched = splitIntoBatches(cropedImages, recognitionBatchSize);
    for (List<Mat> iter : batched) {
      List<List<RecognitionResult>> batchResults = textRecognitionModel.batchPredict(
          iter.stream().map(mat -> new PreProcessResult(mat, null)).toList(),
          recognitionBatchSize, matManager, ndManager, Map.of());
      for (List<RecognitionResult> recResults : batchResults) {
        String textLineOriLabel = (index < textLineOriLabels.size()) ? textLineOriLabels.get(index) : null;
        float textLineOriScore = (index < textLineOriScores.size()) ? textLineOriScores.get(index) : 0f;
        results.add(new OCRPipelineResult(
            polys[index],
            recResults,
            docOriLabel,
            docOriScore,
            textLineOriLabel,
            textLineOriScore
        ));
        index++;
      }
    }
    polysNDArray.close();
    sortedPolysArray.close();
    return results;
  }

}
