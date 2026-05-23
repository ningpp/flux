package io.github.flux.falconocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.ModelInstanceKey;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class FalconOcrModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of("Falcon-OCR");

    private static final int DEFAULT_MAX_SEQ_LEN = 8192;
    private static final Map<ModelInstanceKey, FalconOcrModel> INSTANCE_CACHE = new ConcurrentHashMap<>();

    public static FalconOcrModel getSharedInstance(final String modelRootDir,
                                                   final String modelName,
                                                   final int gpuIndex,
                                                   final OrtEnvironment env,
                                                   final Map<String, Object> customParams) {
        ModelInstanceKey key = new ModelInstanceKey(modelRootDir, modelName, gpuIndex, customParams);
        return INSTANCE_CACHE.computeIfAbsent(key,
                k -> new FalconOcrModel(modelRootDir, modelName, gpuIndex, env, customParams));
    }

    private final HuggingFaceTokenizer tokenizer;
    private final FalconOcrTokenKvModel tokenKvModel;

    private FalconOcrModel(final String modelRootDir,
                           final String modelName,
                           final int gpuIndex,
                           final OrtEnvironment env,
                           final Map<String, Object> customParams) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported Falcon-OCR model: " + modelName);
        }
        final String modelDir = modelRootDir + File.separator + modelName;
        Integer maxNewTokens = getOptionalPositiveInt(customParams, "maxNewTokens");
        int maxSeqLen = readMaxSeqLen(modelDir);
        try {
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.tokenKvModel = new FalconOcrTokenKvModel(
                    new File(modelDir, "falcon_ocr_kv_token.onnx").getAbsolutePath(),
                    gpuIndex,
                    env,
                    maxNewTokens,
                    maxSeqLen
            );
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    List<TextResult> predictCategory(List<PreProcessResult> images,
                                     MatManager matManager,
                                     String category) {
        try {
            List<FalconOcrProcessor.Preprocessed> items = new ArrayList<>(images.size());
            for (PreProcessResult image : images) {
                items.add(FalconOcrProcessor.process(matManager, image.mat(), tokenizer, category));
            }
            FalconOcrProcessor.BatchPreprocessed batch = FalconOcrProcessor.batchPad(items);
            long[][] generatedIds = tokenKvModel.predict(batch);
            List<TextResult> results = new ArrayList<>(generatedIds.length);
            for (long[] ids : generatedIds) {
                results.add(new TextResult(decodeGenerated(ids), ids, -1));
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> images,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        String category = String.valueOf(
                extraParameters == null ? "formula" : extraParameters.getOrDefault("category", "formula"));
        return predictCategory(images, matManager, category);
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(rgbMat, null);
    }

    private String decodeGenerated(long[] ids) {
        return tokenizer.decode(ids, false)
                .replace("<|end_of_query|>", "")
                .replace("<|end_of_text|>", "")
                .strip();
    }

    private static Integer getOptionalPositiveInt(Map<String, Object> customParams, String key) {
        if (customParams == null || !customParams.containsKey(key)) {
            return null;
        }
        Object value = customParams.get(key);
        int parsed;
        if (value instanceof Number number) {
            parsed = number.intValue();
        } else {
            parsed = Integer.parseInt(String.valueOf(value));
        }
        if (parsed <= 0) {
            throw new FluxException("Falcon-OCR " + key + " must be positive, got: " + parsed);
        }
        return parsed;
    }

    private static int readMaxSeqLen(String modelDir) {
        File configFile = new File(modelDir, "config.json");
        try {
            JsonObject config = JsonParser.parseString(
                    Files.readString(configFile.toPath(), StandardCharsets.UTF_8)).getAsJsonObject();
            if (config.has("max_seq_len")) {
                return config.get("max_seq_len").getAsInt();
            }
        } catch (Exception e) {
            throw new FluxException(e);
        }
        return DEFAULT_MAX_SEQ_LEN;
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(tokenKvModel);
        IOUtil.close(tokenizer);
    }
}
