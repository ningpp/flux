package io.github.flux.llava;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.llava.LlavaOneVisionImageProcessor.ImageProcessResult;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * LLaVA-OneVision-Qwen2 model orchestrator.
 * Pipeline: image → vision encoder → embed tokens → merge image features
 *           → compute 2D position_ids → decoder (KV-cache) → text
 *
 * Architecture: SigLIP ViT + Qwen2-0.5B decoder.
 * Unlike Qwen3-VL, this model uses standard 2D RoPE (not MRoPE) and has no deepstack.
 * Image tokens use simple incrementing position IDs.
 */
public class LlavaOneVisionQwenModel extends BatchPredictor<ImageProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "llava-onevision-qwen2-0.5b-ov-hf"
    );

    // Special token IDs from config
    private static final long IM_START = 151644L;
    private static final long IM_END = 151645L;
    private static final long IMAGE_TOKEN = 151646L;

    private final LlavaOneVisionEncoderModel encoderModel;
    private final LlavaOneVisionEmbedModel embedModel;
    private final LlavaOneVisionDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final int hiddenSize;

    public LlavaOneVisionQwenModel(final String modelRootDir,
                                    final String modelName,
                                    final int gpuIndex,
                                    final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported model: " + modelName);
        }
        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.encoderModel = new LlavaOneVisionEncoderModel(
                    new File(modelDir, "onnx" + File.separator + "vision_encoder.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.embedModel = new LlavaOneVisionEmbedModel(
                    new File(modelDir, "onnx" + File.separator + "embed_tokens.onnx").getAbsolutePath(),
                    gpuIndex, env);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));

            // Read model config for dynamic architecture constants
            Type mapType = new TypeToken<Map<String, Object>>() {}.getType();
            Map<String, Object> config = new Gson().fromJson(
                    Files.readString(new File(modelDir, "config.json").toPath(), StandardCharsets.UTF_8),
                    mapType);
            @SuppressWarnings("unchecked")
            Map<String, Object> textConfig = (Map<String, Object>) config.getOrDefault("text_config", Map.of());
            int numLayers = ((Number) textConfig.getOrDefault("num_hidden_layers", 24)).intValue();
            int numKvHeads = ((Number) textConfig.getOrDefault("num_key_value_heads", 2)).intValue();
            int numAttentionHeads = ((Number) textConfig.getOrDefault("num_attention_heads", 14)).intValue();
            this.hiddenSize = ((Number) textConfig.getOrDefault("hidden_size", 896)).intValue();
            int headDim = hiddenSize / numAttentionHeads;  // 896 / 14 = 64

            this.decoderModel = new LlavaOneVisionDecoderModel(
                    new File(modelDir, "onnx" + File.separator + "decoder_model_merged.onnx").getAbsolutePath(),
                    gpuIndex, env, 4096,
                    numLayers, numKvHeads, headDim, this.hiddenSize);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public List<TextResult> doBatchPredict(List<ImageProcessResult> imageResults,
                                           MatManager matManager,
                                           NDManager ndManager,
                                           Map<String, Object> extraParameters) {
        try {
            int batchSize = imageResults.size();
            boolean debug = Boolean.TRUE.equals(extraParameters != null ? extraParameters.get("debug") : false);
            String prompt = extraParameters != null && extraParameters.get("prompt") instanceof String 
                    ? (String) extraParameters.get("prompt") 
                    : "OCR";

            // 1. Vision encode each image
            List<float[][]> encoderResults = new ArrayList<>(batchSize);
            for (ImageProcessResult ir : imageResults) {
                float[][][] pixelValues = new float[1][][];
                pixelValues[0] = ir.pixelValues();
                float[][][] encoded = encoderModel.predict(pixelValues);
                encoderResults.add(encoded[0]);  // [729, hidden_size]
            }

            // All images produce the same token count (729 for SigLIP 384x384)
            int numImageTokens = encoderResults.get(0).length;  // 729
            int visionHiddenSize = encoderResults.get(0)[0].length;  // 1152
            
            if (debug) {
                System.out.println("\n--- Vision Encoder Output ---");
                System.out.println("Image features shape: [" + batchSize + ", " + numImageTokens + ", " + visionHiddenSize + "]");
                System.out.println("Number of image tokens: " + numImageTokens);
                System.out.println("Vision hidden size: " + visionHiddenSize);
                
                float[][] features = encoderResults.get(0);
                System.out.print("First 5 features of first token: [");
                for (int i = 0; i < Math.min(5, features[0].length); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.printf("%.6f", features[0][i]);
                }
                System.out.println("]");
            }

            // 2. Build template input_ids (same for all images)
            long[] templateIds = buildInputIds(numImageTokens, prompt);
            int seqLen = templateIds.length;
            
            if (debug) {
                System.out.println("\n--- Input IDs ---");
                System.out.println("Input IDs shape: [" + batchSize + ", " + seqLen + "]");
                System.out.print("Input IDs: [");
                for (int i = 0; i < Math.min(30, seqLen); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.print(templateIds[i]);
                }
                if (seqLen > 30) System.out.print(", ... (" + (seqLen - 30) + " more)");
                System.out.println("]");
                
                // Decode to show prompt
                String decodedPrompt = tokenizer.decode(templateIds);
                System.out.println("Decoded prompt:\n" + decodedPrompt);
            }

            // 3. Batch embed tokens
            long[][] batchedIds = new long[batchSize][];
            for (int b = 0; b < batchSize; b++) {
                batchedIds[b] = templateIds.clone();
            }
            float[][][] inputsEmbeds = embedModel.predict(batchedIds);

            // 4. Find IMAGE_TOKEN position (should be single token at position ~14)
            int imageTokenPos = -1;
            for (int pos = 0; pos < seqLen; pos++) {
                if (templateIds[pos] == IMAGE_TOKEN) {
                    imageTokenPos = pos;
                    break;
                }
            }
            
            if (imageTokenPos == -1) {
                throw new FluxException("IMAGE_TOKEN not found in prompt");
            }

            // 5. Expand sequence: replace single IMAGE_TOKEN with all vision feature embeddings
            // New sequence length: original - 1 (remove IMAGE_TOKEN) + numImageTokens (add vision features)
            int expandedSeqLen = seqLen - 1 + numImageTokens;
            
            if (debug) {
                System.out.println("\n--- Merging Vision Features ---");
                System.out.println("IMAGE_TOKEN position: " + imageTokenPos);
                System.out.println("Original sequence length: " + seqLen);
                System.out.println("Expanded sequence length: " + expandedSeqLen);
                System.out.println("Vision features to insert: " + numImageTokens);
            }
            
            float[][][] expandedEmbeds = new float[batchSize][expandedSeqLen][hiddenSize];
            
            for (int b = 0; b < batchSize; b++) {
                float[][] features = encoderResults.get(b);
                
                // Copy embeddings before IMAGE_TOKEN
                for (int i = 0; i < imageTokenPos; i++) {
                    System.arraycopy(inputsEmbeds[b][i], 0, expandedEmbeds[b][i], 0, hiddenSize);
                }
                
                // Insert all vision feature embeddings
                for (int i = 0; i < numImageTokens; i++) {
                    System.arraycopy(features[i], 0, expandedEmbeds[b][imageTokenPos + i], 0, hiddenSize);
                }
                
                // Copy embeddings after IMAGE_TOKEN
                for (int i = imageTokenPos + 1; i < seqLen; i++) {
                    int targetPos = imageTokenPos + numImageTokens + (i - imageTokenPos - 1);
                    System.arraycopy(inputsEmbeds[b][i], 0, expandedEmbeds[b][targetPos], 0, hiddenSize);
                }
            }
            
            inputsEmbeds = expandedEmbeds;
            seqLen = expandedSeqLen;
            
            if (debug) {
                System.out.println("Final inputs_embeds shape: [" + batchSize + ", " + seqLen + ", " + hiddenSize + "]");
                
                // Print sample embedding values for comparison
                System.out.print("First 5 embedding values at position 0: [");
                for (int i = 0; i < Math.min(5, hiddenSize); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.printf("%.6f", expandedEmbeds[0][0][i]);
                }
                System.out.println("]");
                
                System.out.print("First 5 embedding values at position " + imageTokenPos + " (first vision feature): [");
                for (int i = 0; i < Math.min(5, hiddenSize); i++) {
                    if (i > 0) System.out.print(", ");
                    System.out.printf("%.6f", expandedEmbeds[0][imageTokenPos][i]);
                }
                System.out.println("]");
            }

            // 6. Compute 2D position_ids (simple incrementing sequence)
            long[][] positionIds = new long[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < seqLen; i++) {
                    positionIds[b][i] = i;
                }
            }

            // 7. Attention mask [batchSize, seqLen]
            long[][] attentionMask = new long[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < seqLen; i++) {
                    attentionMask[b][i] = 1L;
                }
            }

            // 8. Decode (batched)
            long[][] generatedIds = decoderModel.predict(
                    inputsEmbeds, attentionMask, positionIds, embedModel);

            // 9. Decode tokens to text
            List<TextResult> results = new ArrayList<>();
            for (long[] tokens : generatedIds) {
                String text = tokenizer.decode(tokens);
                results.add(new TextResult(text, tokens, -1));
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Build prompt input_ids with the LLaVA chat template.
     * 
     * Format:
     * <|im_start|>system
     * You are a helpful assistant.<|im_end|>
     * <|im_start|>user
     * <image>
     * {prompt}<|im_end|>
     * <|im_start|>assistant
     * 
     * @param numImageTokens Number of image tokens (unused, kept for API compatibility)
     * @param prompt The user prompt/question about the image
     */
    private long[] buildInputIds(int numImageTokens, String prompt) {
        List<Long> ids = new ArrayList<>();
        
        // System message
        addTurnStart(ids, "system", false);  // No leading newline for first turn
        addEncoded(ids, "You are a helpful assistant.");
        addTurnEnd(ids);
        
        // User message with image
        addTurnStart(ids, "user", true);
        ids.add(IMAGE_TOKEN);  // Single token, expanded to 729 features during embedding merge
        addEncoded(ids, prompt);
        addTurnEnd(ids);
        
        // Assistant turn start (generation begins here)
        addTurnStart(ids, "assistant", true);
        
        return ids.stream().mapToLong(Long::longValue).toArray();
    }
    
    private void addTurnStart(List<Long> ids, String role, boolean addNewline) {
        if (addNewline) {
            addEncoded(ids, "\n");
        }
        ids.add(IM_START);
        addEncoded(ids, role + "\n");
    }
    
    private void addTurnEnd(List<Long> ids) {
        ids.add(IM_END);
    }
    
    private void addEncoded(List<Long> ids, String text) {
        for (long id : tokenizer.encode(text).getIds()) {
            ids.add(id);
        }
    }

    @Override
    public ImageProcessResult processRgb(MatManager matManager, Mat rgbMat, NDManager ndManager) {
        return LlavaOneVisionImageProcessor.process(rgbMat, matManager);
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(embedModel);
        IOUtil.close(tokenizer);
    }
}
