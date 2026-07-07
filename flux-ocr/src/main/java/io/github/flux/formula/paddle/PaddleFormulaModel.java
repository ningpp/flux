package io.github.flux.formula.paddle;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class PaddleFormulaModel extends BatchPredictor<PreProcessResult, TextResult> {

  public static final List<String> MODEL_NAMES = List.of(
      "PP-FormulaNet-L",
      "PP-FormulaNet_plus-L"
  );

  static {
    FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, PaddleFormulaModel::new);
  }

  private final OrtEnvironment env;
  private final OrtSession encoderSession;
  private final OrtSession decoderSession;
  private final HuggingFaceTokenizer tokenizer;

  public PaddleFormulaModel(
      final String modelRootDir,
      final String modelName,
      final int gpuIndex,
      final OrtEnvironment env,
      final Map<String, Object> customParams) {
    try {
      String modelDir = modelRootDir + File.separator + modelName;
      this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
      this.env = env;
      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      if (gpuIndex > -1) {
        options.addCUDA(gpuIndex);
      }
      this.encoderSession = env.createSession(
          new File(modelDir, "encoder.onnx").getAbsolutePath(), options);
      this.decoderSession = env.createSession(
          new File(modelDir, "decoder_model_merged.onnx").getAbsolutePath(), options);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public List<TextResult> doBatchPredict(List<PreProcessResult> mats, MatManager matManager, NDManager ndManager, Map<String, Object> extraParameters) {
    long eos_token_id = 2;
    long decoder_start_token_id = eos_token_id;
    int max_length = 1537;
    int layers = 8;
    int heads = 16;
    int headDim = 32;
    int batchSize = mats.size();
    Result encoderResult = null;
    Result activeKvResult = null;
    List<OnnxTensor> initialPkvSelf = null;
    boolean initialPkvSelfClosed = false;
    try {
      OnnxTensor pv = createPixelValuesTensor(mats);
      // 预处理 NDArray 在拼入像素张量后即不再需要，立即释放，避免其滞留于
      // （可能长期存活的）NDManager 中逐步累积 -> 内存泄露。
      for (PreProcessResult ppr : mats) {
        IOUtil.close(ppr);
      }
      try {
        encoderResult = encoderSession.run(Map.of("pv", pv));
      } finally {
        IOUtil.close(pv);
      }
      OnnxTensor encodeResult = (OnnxTensor) encoderResult.get(0);

      long[] currentTokens = new long[batchSize];
      Arrays.fill(currentTokens, decoder_start_token_id);

      long[][] generatedTokens = new long[batchSize][];
      boolean[] finished = new boolean[batchSize];
      for (int i = 0; i < batchSize; i++) {
        generatedTokens[i] = new long[] {decoder_start_token_id};
      }

      initialPkvSelf = createEmptySelfKv(batchSize, layers, heads, headDim);
      List<OnnxTensor> pkv_self = initialPkvSelf;
      List<String> kn_self = kv_names_self("past", layers);
      int max = max_length - 1;
      for (int step = 0; step < max; step++) {
        if (ArrayUtil.allTrue(finished)) {
          break;
        }
        Map<String, OnnxTensor> fd = new HashMap<>();
        OnnxTensor stepIds = createTokenTensor(currentTokens, finished, eos_token_id);
        fd.put("ids", stepIds);
        fd.put("enc", encodeResult);
        for (int i = 0; i < kn_self.size(); i++) {
          fd.put(kn_self.get(i), pkv_self.get(i));
        }

        Result decodeResult = null;
        try {
          decodeResult = decoderSession.run(fd);
          long[] nextTokens = readTokenBatch((OnnxTensor) decodeResult.get(0), batchSize);
          List<OnnxTensor> nextPkvSelf = extractSelfKv(decodeResult, layers);

          if (activeKvResult == null) {
            closeTensors(initialPkvSelf);
            initialPkvSelfClosed = true;
          } else {
            IOUtil.close(activeKvResult);
          }
          activeKvResult = decodeResult;
          decodeResult = null;
          pkv_self = nextPkvSelf;

          for (int i = 0; i < batchSize; i++) {
            if (finished[i]) {
              currentTokens[i] = eos_token_id;
              continue;
            }
            currentTokens[i] = nextTokens[i];
            generatedTokens[i] = ArrayUtil.concat(generatedTokens[i], new long[] {nextTokens[i]});
            if (nextTokens[i] == eos_token_id) {
              finished[i] = true;
            }
          }
        } finally {
          IOUtil.close(stepIds);
          IOUtil.close(decodeResult);
        }
      }

      List<TextResult> textResults = new ArrayList<>();
      for (long[] tokens : generatedTokens) {
        textResults.add(new TextResult(tokenizer.decode(tokens, true), tokens, -1));
      }
      return textResults;
    } catch (Exception e) {
      throw new RuntimeException(e);
    } finally {
      if (!initialPkvSelfClosed) {
        closeTensors(initialPkvSelf);
      }
      IOUtil.close(activeKvResult);
      IOUtil.close(encoderResult);
    }
  }

  private static final List<String> NAMES2 = List.of("sk", "sv");

  private OnnxTensor createPixelValuesTensor(List<PreProcessResult> mats) throws Exception {
    long[] imageShape = mats.get(0).ndArray().getShape().getShape();
    long imageSize = 1;
    for (long dim : imageShape) {
      imageSize *= dim;
    }
    int batchSize = mats.size();
    float[] data = new float[Math.toIntExact(imageSize * batchSize)];
    for (int i = 0; i < batchSize; i++) {
      long[] shape = mats.get(i).ndArray().getShape().getShape();
      if (!Arrays.equals(imageShape, shape)) {
        throw new IllegalArgumentException("All pixel values must have the same shape");
      }
      float[] imageData = mats.get(i).ndArray().toFloatArray();
      if (imageData.length != imageSize) {
        throw new IllegalArgumentException("Unexpected pixel value size: " + imageData.length);
      }
      System.arraycopy(imageData, 0, data, Math.toIntExact(imageSize * i), imageData.length);
    }
    return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), ArrayUtil.concat(new long[] {batchSize}, imageShape));
  }

  private OnnxTensor createTokenTensor(long[] tokenIds, boolean[] finished, long eosTokenId) throws Exception {
    long[][] data = new long[tokenIds.length][1];
    for (int i = 0; i < tokenIds.length; i++) {
      data[i][0] = finished[i] ? eosTokenId : tokenIds[i];
    }
    return ArrayUtil.createOnnxTensor(data, env);
  }

  private long[] readTokenBatch(OnnxTensor tokenTensor, int batchSize) throws Exception {
    Object value = tokenTensor.getValue();
    long[] tokens = new long[batchSize];
    if (value instanceof long[][][] array) {
      for (int i = 0; i < batchSize; i++) {
        tokens[i] = array[i][0][0];
      }
    } else if (value instanceof long[][] array) {
      for (int i = 0; i < batchSize; i++) {
        tokens[i] = array[i][0];
      }
    } else if (value instanceof long[] array) {
      System.arraycopy(array, 0, tokens, 0, batchSize);
    } else {
      throw new IllegalStateException("Unsupported token tensor type: " + value.getClass());
    }
    return tokens;
  }

  private List<OnnxTensor> extractSelfKv(Result result, int layers) {
    List<OnnxTensor> kvSelf = new ArrayList<>();
    for (int i = 0; i < layers * NAMES2.size(); i++) {
      kvSelf.add((OnnxTensor) result.get(i + 1));
    }
    return kvSelf;
  }

  private List<String> kv_names_self(String pref, int n) {
    List<String> r = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      for (String name : NAMES2) {
        r.add(String.format(Locale.ROOT, "%s_%d_%s", pref, i, name));
      }
    }
    return r;
  }

  private List<OnnxTensor> createEmptySelfKv(int batchSize, int layers, int heads, int headDim) throws Exception {
    List<OnnxTensor> tensors = new ArrayList<>();
    for (int i = 0; i < layers * NAMES2.size(); i++) {
      tensors.add(OnnxTensor.createTensor(
          env,
          FloatBuffer.wrap(new float[0]),
          new long[] {batchSize, heads, 0, headDim}
      ));
    }
    return tensors;
  }

  private void closeTensors(List<OnnxTensor> tensors) {
    if (tensors == null) {
      return;
    }
    for (OnnxTensor tensor : tensors) {
      IOUtil.close(tensor);
    }
  }

  @Override
  public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
    var margined = cropMargin(rgbMat, matManager, 200);
    var resized = resize(margined, matManager, 768, false, null);
    var thumbnailed = thumbnail(resized, matManager, 768, 768);
    var padded = padImage(thumbnailed, matManager, 768, 768);
    var rescaled = rescaleAndNormalize(padded, matManager, true, 0.00392156862745098,
        true, new double[] {0.7931, 0.7931, 0.7931},
        new double[]{0.1738, 0.1738, 0.1738});
    var ndarray = ImageUtil.toNDArrayFloat(rescaled, ndManager);
    var transposed = ndarray.transpose(2, 0, 1);
    // transpose 返回的是 ndarray 的视图，其底层 base(ndarray) 在视图关闭时不会自动释放，
    // 会残留于 NDManager（内存泄露）。duplicate 出独立副本后关闭视图与 base，
    // 确保仅返回 1 个独立 NDArray。Mat 已全部在使用后释放。
    var resultNd = transposed.duplicate();
    transposed.close();
    ndarray.close();
    matManager.release(margined);
    matManager.release(resized);
    matManager.release(thumbnailed);
    matManager.release(padded);
    matManager.release(rescaled);
    matManager.release(rgbMat);
    return new PreProcessResult(null, resultNd);
  }

  @Override
  public void close() throws Exception {
    IOUtil.close(tokenizer);
    IOUtil.close(encoderSession);
    IOUtil.close(decoderSession);
  }

  public static Mat normalize(
      Mat image,
      MatManager matManager,
      double[] mean,
      double[] std
  ) {

    Mat result = matManager.newMat();

    image.convertTo(result, CvType.CV_32F);

    List<Mat> channels = matManager.split(result);

    for (int i = 0; i < channels.size(); i++) {

      Core.subtract(
          channels.get(i),
          new Scalar(mean[i]),
          channels.get(i)
      );

      Core.divide(
          channels.get(i),
          new Scalar(std[i]),
          channels.get(i)
      );
    }

    Mat normalized = matManager.newMat();
    Core.merge(channels, normalized);

    // split 产生的 3 个单通道 Mat 仅用于逐通道减均值/除标准差，merge 后不再需要，
    // 必须释放，否则它们会残留在 MatManager 跟踪表中（此前每轮泄漏 3 个 Mat）。
    matManager.releaseAll(channels);
    // result 仅在 split 前作为中间缓冲，split 后不再需要，同样必须释放（每轮泄漏 1 个 Mat）。
    matManager.release(result);

    return normalized;
  }

  public static Mat rescale(
      Mat image,
      MatManager matManager,
      double scale
  ) {

    Mat result = matManager.newMat();

    image.convertTo(
        result,
        CvType.CV_32F,
        scale
    );

    return result;
  }

  public static FuseResult fuseMeanStdAndRescaleFactor(
      boolean doNormalize,
      double[] imageMean,
      double[] imageStd,
      boolean doRescale,
      double rescaleFactor
  ) {

    if (doRescale && doNormalize) {

      double[] fusedMean = new double[imageMean.length];
      double[] fusedStd = new double[imageStd.length];

      for (int i = 0; i < imageMean.length; i++) {

        fusedMean[i] = imageMean[i] / rescaleFactor;
        fusedStd[i] = imageStd[i] / rescaleFactor;
      }

      doRescale = false;

      return new FuseResult(
          fusedMean,
          fusedStd,
          false
      );
    }

    return new FuseResult(
        imageMean,
        imageStd,
        doRescale
    );
  }
  public static Mat rescaleAndNormalize(
      Mat image,
      MatManager matManager,
      boolean doRescale,
      double rescaleFactor,
      boolean doNormalize,
      double[] imageMean,
      double[] imageStd
  ) {

    FuseResult fused = fuseMeanStdAndRescaleFactor(
        doNormalize,
        imageMean,
        imageStd,
        doRescale,
        rescaleFactor
    );

    imageMean = fused.mean;
    imageStd = fused.std;
    doRescale = fused.doRescale;

    if (doNormalize) {

      return normalize(
          image,
          matManager,
          imageMean,
          imageStd
      );

    } else if (doRescale) {

      return rescale(
          image,
          matManager,
          rescaleFactor
      );
    }

    return image;
  }
  public static class FuseResult {

    public double[] mean;
    public double[] std;
    public boolean doRescale;

    public FuseResult(
        double[] mean,
        double[] std,
        boolean doRescale
    ) {
      this.mean = mean;
      this.std = std;
      this.doRescale = doRescale;
    }
  }

  /**
   * 居中 padding 到指定尺寸
   */
  public static Mat padImage(
      Mat image,
      MatManager matManager,
      int outputHeight,
      int outputWidth
  ) {

    int inputHeight = image.rows();
    int inputWidth = image.cols();

    int deltaWidth = outputWidth - inputWidth;
    int deltaHeight = outputHeight - inputHeight;

    int padTop = deltaHeight / 2;
    int padLeft = deltaWidth / 2;

    int padBottom = deltaHeight - padTop;
    int padRight = deltaWidth - padLeft;

    Mat padded = matManager.newMat();

    // 默认补黑边
    Core.copyMakeBorder(
        image,
        padded,
        padTop,
        padBottom,
        padLeft,
        padRight,
        Core.BORDER_CONSTANT,
        new Scalar(0, 0, 0)
    );

    return padded;
  }

  public static Mat thumbnail(
      Mat image,
      MatManager matManager,
      int outputHeight,
      int outputWidth
  ) {

    int inputHeight = image.rows();
    int inputWidth = image.cols();

    int height = Math.min(inputHeight, outputHeight);
    int width = Math.min(inputWidth, outputWidth);

    // 不需要 resize
    if (height == inputHeight && width == inputWidth) {
      return image;
    }

    if (inputHeight > inputWidth) {

      width = (int)(
          inputWidth * (double) height / inputHeight
      );

    } else if (inputWidth > inputHeight) {

      height = (int)(
          inputHeight * (double) width / inputWidth
      );
    }

    Mat resized = matManager.newMat();

    Imgproc.resize(
        image,
        resized,
        new Size(width, height),
        0,
        0,
        Imgproc.INTER_CUBIC
    );

    return resized;
  }

  /**
   * 对齐 torchvision get_resize_output_image_size
   */
  public static Size getResizeOutputImageSize(
      Mat image,
      MatManager matManager,
      int size,
      boolean defaultToSquare,
      Integer maxSize
  ) {

    int width = image.cols();
    int height = image.rows();

    // square resize
    if (defaultToSquare) {
      return new Size(size, size);
    }

    int shortEdge;
    int longEdge;

    boolean widthIsShorter = width <= height;

    if (widthIsShorter) {
      shortEdge = width;
      longEdge = height;
    } else {
      shortEdge = height;
      longEdge = width;
    }

    int requestedNewShort = size;

    int newShort = requestedNewShort;

    int newLong = (int) (
        requestedNewShort * (double) longEdge / shortEdge
    );

    // max_size logic
    if (maxSize != null) {

      if (maxSize <= requestedNewShort) {
        throw new IllegalArgumentException(
            "maxSize must be > size"
        );
      }

      if (newLong > maxSize) {

        newShort = (int) (
            maxSize * (double) newShort / newLong
        );

        newLong = maxSize;
      }
    }

    int outWidth;
    int outHeight;

    if (widthIsShorter) {
      outWidth = newShort;
      outHeight = newLong;
    } else {
      outWidth = newLong;
      outHeight = newShort;
    }

    return new Size(outWidth, outHeight);
  }

  /**
   * torchvision 风格 resize
   */
  public static Mat resize(
      Mat image,
      MatManager matManager,
      int size,
      boolean defaultToSquare,
      Integer maxSize
  ) {

    Size newSize = getResizeOutputImageSize(
        image,
        matManager,
        size,
        defaultToSquare,
        maxSize
    );

    Mat resized = matManager.newMat();

    Imgproc.resize(
        image,
        resized,
        newSize,
        0,
        0,
        Imgproc.INTER_CUBIC
    );

    return resized;
  }

  /**
   * 裁剪图像边缘灰色区域
   *
   * @param image RGB图像
   * @param grayThreshold 灰度阈值，小于该值的像素认为是有效区域
   * @return 裁剪后的图像
   */
  public static Mat cropMargin(Mat image, MatManager matManager, int grayThreshold) {

    // 转灰度
    Mat gray = matManager.newMat();
    Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

    // 最大最小值
    Core.MinMaxLocResult mm = Core.minMaxLoc(gray);

    double minVal = mm.minVal;
    double maxVal = mm.maxVal;

    // 整张图一样
    if (maxVal == minVal) {
      matManager.release(gray);
      return image;
    }

    // 归一化到 0~255
    Mat normalized = matManager.newMat();

    double scale = 255.0 / (maxVal - minVal);
    double shift = -minVal * scale;

    // dst = src * scale + shift
    gray.convertTo(
        normalized,
        CvType.CV_8U,
        scale,
        shift
    );

    // 小于阈值 => 白色
    Mat mask = matManager.newMat();

    Imgproc.threshold(
        normalized,
        mask,
        grayThreshold,
        255,
        Imgproc.THRESH_BINARY_INV
    );

    // 查找非零点
    Mat nonZero = matManager.newMat();
    Core.findNonZero(mask, nonZero);

    // 没有内容
    if (nonZero.empty()) {
      matManager.release(gray);
      matManager.release(normalized);
      matManager.release(mask);
      matManager.release(nonZero);
      return image;
    }

    // bounding rect
    Rect rect = Imgproc.boundingRect(new MatOfPoint(nonZero));

    // crop
    Mat cropped = matManager.newMat(image, rect);
    // gray/normalized/mask/nonZero 仅为求 bounding rect 的临时 Mat，用完即释放，
    // 否则每个 cropMargin 调用会残留 4 个 Mat（此前每轮泄漏的主要来源）。
    matManager.release(gray);
    matManager.release(normalized);
    matManager.release(mask);
    matManager.release(nonZero);
    return cropped;
  }

}
