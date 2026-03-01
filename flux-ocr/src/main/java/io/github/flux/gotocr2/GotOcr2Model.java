package io.github.flux.gotocr2;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class GotOcr2Model extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "GOT-OCR-2.0"
    );

    private final GotOcr2EncoderModel encoderModel;
    private final GotOcr2EmbedModel embedModel;
    private final GotOcr2DecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;

    static {
        FormulaRecognitionModel.getRegistry().register(MODEL_NAMES, GotOcr2Model::new);
    }

    public GotOcr2Model(final String modelRootDir,
                        final String modelName,
                        final int gpuIndex,
                        final OrtEnvironment env,
                        final Map<String, Object> customParams) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new GotOcr2EncoderModel(new File(modelDir, "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.embedModel = new GotOcr2EmbedModel(new File(modelDir, "embed_tokens.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new GotOcr2DecoderModel(
                    new File(modelDir, "decoder_model.onnx").getAbsolutePath(),
                    gpuIndex,
                    env,
                    4096
            );
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> pprs, MatManager matManager, NDManager ndManager, Map<String, Object> extraParameters) {
        final long image_token_index = 151859;
        String imgpadStr = "<imgpad>".repeat(256);
        String prompt = "<|im_start|>system\n"
                + "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
                + "<img>" + imgpadStr + "</img>\n"
                + " OCR with format: <|im_end|><|im_start|>assistant\n";
        long[] one_input_ids = tokenizer.encode(prompt).getIds();
        long[][] inputIds = new long[pprs.size()][];
        for (int i = 0; i < pprs.size(); i++) {
            inputIds[i] = one_input_ids;
        }
        try {
            float[][][] image_features = encoderModel.predict(PreProcessResult.getNDArrays(pprs));
            float[][][] inputs_embeds = embedModel.predict(inputIds);
            prepare_inputs_embeds(inputIds, image_token_index, image_features, inputs_embeds);

            int batchSize = pprs.size();
            int seqLen = inputIds[0].length;

            // attention_mask: [1, seq_len], 全 1
            long[][] attentionMask = new long[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < seqLen; i++) {
                    attentionMask[b][i] = 1L;
                }
            }

            // position_ids: [1, seq_len] -> 0,1,2,...
            long[][] positionIds = new long[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < seqLen; i++) {
                    positionIds[b][i] = i;
                }
            }

                long[][] genIds = decoderModel.predict(image_features,
                    inputIds, inputs_embeds,
                    attentionMask, positionIds, embedModel);
            List<TextResult> textResults = new ArrayList<>();
            for (long[] tokens : genIds) {
                String text = tokenizer.decode(tokens);
                textResults.add(new TextResult(text, tokens, -1));
            }
            return textResults;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private void prepare_inputs_embeds(long[][] inputIds, long image_token_index,
                                       float[][][] image_features,
                                       float[][][] inputs_embeds) {
        int batchSize = inputIds.length;
        int seqLen = inputIds[0].length;

        for (int i = 0; i < batchSize; i++) {
            int imageFeatureIdx = 0; // 对应 Python 里的 j

            for (int pos = 0; pos < seqLen; pos++) {
                if (inputIds[i][pos] == image_token_index) {
                    // inputs_embeds[i, pos] = image_features[i, j]
                    inputs_embeds[i][pos] = image_features[i][imageFeatureIdx];
                    imageFeatureIdx++;
                }
            }
        }
    }

    @Override
    public PreProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return new PreProcessResult(null, GotOcr2ImageProcessor.process(rgbMat, matManager, ndManager));
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}
