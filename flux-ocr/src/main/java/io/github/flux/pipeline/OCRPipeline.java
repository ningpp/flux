package io.github.flux.pipeline;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.RecognitionResult;
import io.github.flux.model.DocOrientationClassifyModel;
import io.github.flux.model.TextDetectionModel;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.model.TextRecognitionModel;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.ParameterUtil;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.flux.util.ArrayUtil.splitIntoBatches;

public class OCRPipeline {

  private final TextDetectionModel textDetectionModel;
  private final TextRecognitionModel textRecognitionModel;
  private final DocOrientationClassifyModel docOriClassifyModel;
  private final TextLineOrientationModel textLineOrientationModel;

  public OCRPipeline(TextDetectionModel textDetectionModel, TextRecognitionModel textRecognitionModel, DocOrientationClassifyModel docOriClassifyModel, TextLineOrientationModel textLineOrientationModel) {
    this.textDetectionModel = textDetectionModel;
    this.textRecognitionModel = textRecognitionModel;
    this.docOriClassifyModel = docOriClassifyModel;
    this.textLineOrientationModel = textLineOrientationModel;
  }

  public List<OCRPipelineResult> predictFile(String img, Map<String, Object> extraParameters) {
    try (
        NDManager ndManager = NDManager.newBaseManager();
        MatManager matManager = new MatManager();
    ) {
      Integer recognitionBatchSize = ParameterUtil.getInteger(extraParameters, "recognitionBatchSize");
      if (recognitionBatchSize == null || recognitionBatchSize < 1) {
        recognitionBatchSize = 1;
      }
      List<OCRPipelineResult> ocrPipelineResults = new ArrayList<>();
      Mat bgrImage = matManager.imread(img, Imgcodecs.IMREAD_COLOR_BGR);
      Mat rgbImg = matManager.newMat();
      Imgproc.cvtColor(bgrImage, rgbImg, Imgproc.COLOR_BGR2RGB);

      // Doc orientation classification (optional)
      String oriLabel = null;
      float oriScore = 0f;
      Mat srcImage;
      if (docOriClassifyModel != null) {
        ClassificationResult oriClassifyResult = docOriClassifyModel.batchPredict(
            List.of(docOriClassifyModel.processRgb(matManager, rgbImg, ndManager)), 1,
            matManager, ndManager, extraParameters).get(0);
        oriLabel = oriClassifyResult.label();
        oriScore = oriClassifyResult.score();
        if (oriScore > 0.3f) {
          srcImage = ImageUtil.rotateImage(matManager, bgrImage, Double.parseDouble(oriLabel));
          rgbImg.release();
        } else {
          srcImage = bgrImage;
        }
      } else {
        srcImage = bgrImage;
      }

      var detectionResult = textDetectionModel.batchPredict(
          List.of(new PreProcessResult(matManager.cloneMat(srcImage), null)), 1,
          matManager, ndManager, extraParameters).get(0);

      int[][][] polys = detectionResult.polys();
      NDArray polysNDArray = ArrayUtil.int3dToNDArray(ndManager, polys);
      NDArray sortedPolysArray = new SortQuadBoxes().sort(ndManager, polysNDArray);
      int[][][] sortedPolys = ArrayUtil.toInt3d(sortedPolysArray);
      List<Mat> cropedImages = new ArrayList<>();

      for (int[][] poly : sortedPolys) {
        cropedImages.add(ImageUtil.getMinAreaRectCrop(matManager, ndManager, srcImage, poly));
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
        List<List<RecognitionResult>> batchResults = textRecognitionModel
            .batchPredict(iter.stream().map(mat->new PreProcessResult(mat, null)).toList(),
                recognitionBatchSize, matManager, ndManager, Map.of());
        for (List<RecognitionResult> recResults : batchResults) {
          String textLineOriLabel = (index < textLineOriLabels.size()) ? textLineOriLabels.get(index) : null;
          float textLineOriScore = (index < textLineOriScores.size()) ? textLineOriScores.get(index) : 0f;
          ocrPipelineResults.add(new OCRPipelineResult(
              polys[index],
              recResults,
              oriLabel,
              oriScore,
              textLineOriLabel,
              textLineOriScore
          ));
          index++;
        }
      }
      srcImage.release();
      bgrImage.release();
      rgbImg.release();
      polysNDArray.close();
      sortedPolysArray.close();
      return ocrPipelineResults;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

}
