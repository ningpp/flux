package io.github.flux.blip;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class BlipModel extends BatchPredictor<BlipModel.BlipPreProcessResult, TextResult> {

    private final BlipVisionEncoder visionEncoder;
    private final BlipTextDecoder textDecoder;
    private final HuggingFaceTokenizer tokenizer;

    private static final long BOS_TOKEN = 30522L;
    private static final long SEP_TOKEN = 102L;
    private static final int MAX_NEW_TOKENS = 512;

    public BlipModel(String modelRootDir, String modelName, int gpuIndex, OrtEnvironment env) {
        String modelDir = new File(modelRootDir, modelName).getAbsolutePath();
        try {
            this.visionEncoder = new BlipVisionEncoder(new File(modelDir, "blip_vision_encoder.onnx").getAbsolutePath(), gpuIndex, env);
            this.textDecoder = new BlipTextDecoder(new File(modelDir, "blip_text_decoder.onnx").getAbsolutePath(), gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * BLIP 预处理结果：仅持有一份归一化像素 float[]（形状 [C,H,W]），
     * 不持有 DJL NDArray，从而彻底避免 NDArray 原生内存泄漏与冗余的来回拷贝。
     */
    public record BlipPreProcessResult(float[] pixelData) {
    }

    @Override
    public BlipPreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        float[] pixels = BlipImageProcessor.process(rgbMat, matManager);
        // rgbMat 仅用于生成像素数据，使用完毕立即释放，避免 MatManager 跟踪表随推理张数增长（内存泄露）
        matManager.release(rgbMat);
        return new BlipPreProcessResult(pixels);
    }

    @Override
    public List<TextResult> doBatchPredict(List<BlipPreProcessResult> pprs, MatManager matManager, NDManager ndManager, Map<String, Object> extraParameters) {
        int batchSize = pprs.size();
        if (batchSize == 0) return List.of();

        // 1. Prepare Vision Input
        // Aggregate [Batch, 3, 384, 384]
        int channel = 3;
        int height = 384;
        int width = 384;
        int singleImageSize = channel * height * width;
        float[] pixelValues = new float[batchSize * singleImageSize];

        for (int i = 0; i < batchSize; i++) {
            float[] imgData = pprs.get(i).pixelData();
            System.arraycopy(imgData, 0, pixelValues, i * singleImageSize, singleImageSize);
        }

        try {
            // 2. Vision Encoder
            float[][][] encoderHiddenStates = visionEncoder.predict(pixelValues, batchSize);

            // 3. Text Decoder Loop
            // 预分配固定长度缓冲，序列逐步增长，避免每步重新分配“整段已生成前缀”（O(n^2) 拷贝）
            int maxLen = MAX_NEW_TOKENS + 1;
            long[][] sequences = new long[batchSize][maxLen];
            for (int i = 0; i < batchSize; i++) {
                sequences[i][0] = BOS_TOKEN;
            }

            boolean[] finished = new boolean[batchSize];
            int curLen = 1;

            for (int step = 0; step < MAX_NEW_TOKENS; step++) {
                if (allFinished(finished)) break;

                // 仅用前 curLen 列构造输入，复用预分配缓冲
                float[][][] logits = textDecoder.predict(sequences, curLen, encoderHiddenStates);

                for (int b = 0; b < batchSize; b++) {
                    if (finished[b]) {
                        sequences[b][curLen] = SEP_TOKEN;
                    } else {
                        // logits[b] 形状: [curLen, vocabSize]，取最后一个 token 的 logits
                        float[] lastTokenLogits = logits[b][curLen - 1];
                        int maxIdx = argmax(lastTokenLogits);

                        if (maxIdx == SEP_TOKEN) {
                            finished[b] = true;
                        }
                        sequences[b][curLen] = maxIdx;
                    }
                }
                curLen++;
            }

            // 4. Decode
            List<TextResult> results = new ArrayList<>();
            for (int b = 0; b < batchSize; b++) {
                long[] ids = Arrays.copyOf(sequences[b], curLen);
                String text = tokenizer.decode(ids, true);
                results.add(new TextResult(text, ids, 1.0f));
            }
            return results;

        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private boolean allFinished(boolean[] finished) {
        for (boolean b : finished) {
            if (!b) return false;
        }
        return true;
    }

    private int argmax(float[] arr) {
        int best = -1;
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                best = i;
            }
        }
        return best;
    }

    @Override
    public void close() throws Exception {
        if (visionEncoder != null) visionEncoder.close();
        if (textDecoder != null) textDecoder.close();
        // 释放 tokenizer 持有的原生资源（词表/分词器句柄），否则长期持有会导致内存泄漏
        if (tokenizer != null) tokenizer.close();
    }
}
