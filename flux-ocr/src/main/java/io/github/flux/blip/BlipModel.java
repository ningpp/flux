package io.github.flux.blip;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BlipModel extends BatchPredictor<PreProcessResult, TextResult> {

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

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        NDArray array = BlipImageProcessor.process(rgbMat, matManager, ndManager);
        return new PreProcessResult(null, array);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> pprs, MatManager matManager, NDManager ndManager, Map<String, Object> extraParameters) {
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
            float[] imgData = pprs.get(i).ndArray().toFloatArray();
            System.arraycopy(imgData, 0, pixelValues, i * singleImageSize, singleImageSize);
        }

        try {
            // 2. Vision Encoder
            float[][][] encoderHiddenStates = visionEncoder.predict(pixelValues, batchSize);

            // 3. Text Decoder Loop
            long[][] inputIds = new long[batchSize][1];
            for (int i = 0; i < batchSize; i++) inputIds[i][0] = BOS_TOKEN;

            boolean[] finished = new boolean[batchSize];

            for (int step = 0; step < MAX_NEW_TOKENS; step++) {
                if (allFinished(finished)) break;

                float[][][] logits = textDecoder.predict(inputIds, encoderHiddenStates);

                int seqLen = inputIds[0].length;
                long[][] nextInputIds = new long[batchSize][seqLen + 1];

                for (int b = 0; b < batchSize; b++) {
                    System.arraycopy(inputIds[b], 0, nextInputIds[b], 0, seqLen);

                    if (finished[b]) {
                        nextInputIds[b][seqLen] = SEP_TOKEN;
                    } else {
                        // Logits shape: [seqLen, vocabSize] for this batch
                        float[] lastTokenLogits = logits[b][seqLen - 1];
                        int maxIdx = argmax(lastTokenLogits);

                        if (maxIdx == SEP_TOKEN) {
                            finished[b] = true;
                        }
                        nextInputIds[b][seqLen] = maxIdx;
                    }
                }
                inputIds = nextInputIds;
            }

            // 4. Decode
            List<TextResult> results = new ArrayList<>();
            for (int b = 0; b < batchSize; b++) {
                String text = tokenizer.decode(inputIds[b], true);
                results.add(new TextResult(text, inputIds[b], 1.0f));
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
    }
}
