package io.github.flux.dolphin;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ByteDanceDolphinElementModel extends BatchPredictor<PreProcessResult, TextResult> {

    public static final Set<String> MODEL_NAMES = Set.of(
            "Dolphin",
            "Dolphin-1.5"
    );

    private final DolphinEncoderModel encoderModel;
    private final DolphinDecoderModel decoderModel;
    private final HuggingFaceTokenizer tokenizer;
    private final DolphinPreProcessor preProcessor;
    private final boolean skipSpecialTokens;

    public ByteDanceDolphinElementModel(final String modelRootDir,
                                        final String modelName,
                                        final int gpuIndex,
                                        final OrtEnvironment env,
                                        final OnnxJavaType dtype,
                                        final boolean skipSpecialTokens) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported pix2text model: " + modelName);
        }

        final String modelDir = modelRootDir + File.separator + modelName;
        try {
            this.skipSpecialTokens = skipSpecialTokens;
            this.preProcessor = new DolphinPreProcessor();
            String suffix;
            if (dtype == OnnxJavaType.FLOAT) {
                suffix = "_float32";
            } else {
                suffix = "_float16";
            }
            this.encoderModel = new DolphinEncoderModel(new File(modelDir, "encoder_model" + suffix + ".onnx").getAbsolutePath(),
                    gpuIndex, env, dtype);
            this.tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir));
            this.decoderModel = new DolphinDecoderModel(new File(modelDir, "decoder_model" + suffix + ".onnx").getAbsolutePath(),
                    gpuIndex, env, 4096, 1, 2, tokenizer, skipSpecialTokens, dtype);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        return new PreProcessResult(preProcessor.process(rgbMat), null);
    }

    @Override
    public List<TextResult> doBatchPredict(List<PreProcessResult> images, NDManager manager, Map<String, Object> extraParameters) {
        String prompt = String.valueOf(extraParameters.getOrDefault("prompt", "Read text in the image."));
        try {
            String task_prompt = "<s>" + prompt + " <Answer/>";
            long[] decoder_input_ids = tokenizer.encode(task_prompt, false, false).getIds();
            float[][][] encoderResults = encoderModel.batchPredict(PreProcessResult.getMats(images), manager);

            return decoderModel.predict(prompt, encoderResults, decoder_input_ids, manager);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(encoderModel);
        IOUtil.close(decoderModel);
        IOUtil.close(tokenizer);
    }

}
