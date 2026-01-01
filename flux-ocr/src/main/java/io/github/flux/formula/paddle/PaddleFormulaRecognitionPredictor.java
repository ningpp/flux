package io.github.flux.formula.paddle;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.BatchPredictor;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.core.PreProcessResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class PaddleFormulaRecognitionPredictor extends BatchPredictor<PreProcessResult, FormulaRecognitionResult> {
    public static final List<String> MODEL_NAMES = List.of(
            "PP-FormulaNet-S",
            "PP-FormulaNet-L",
            "PP-FormulaNet_plus-S",
            "PP-FormulaNet_plus-M",
            "PP-FormulaNet_plus-L"
    );
    private static final Logger LOGGER = LoggerFactory.getLogger(PaddleFormulaRecognitionPredictor.class);

    private final Set<String> inputNames;
    private final Set<String> outputNames;
    private final UniMERNetImgDecode uniMERNetImgDecode;
    private final UniMERNetTestTransform uniMERNetTestTransform;
    private final LatexImageFormat latexImageFormat;
    private final UniMERNetDecode postProcessor;
    private final OrtEnvironment env;
    private final OrtSession session;

    @Override
    public void close() throws Exception {
        session.close();
    }

    public PaddleFormulaRecognitionPredictor(final String modelRootDir,
                                             final String modelName,
                                             final int gpuIndex,
                                             final OrtEnvironment env) {
        if (!MODEL_NAMES.contains(modelName)) {
            throw new FluxException("not supported paddle model: " + modelName);
        }

        String modelDir = modelRootDir + File.separator + modelName;
        this.env = env;
        try {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                LOGGER.warn("paddle formula model does not support cuda now!");
            }
            this.session = env.createSession(new File(modelDir, "model.onnx").getAbsolutePath(), options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
        this.inputNames = session.getInputNames();
        this.outputNames = session.getOutputNames();

        Size size;
        if (Set.of("PP-FormulaNet-S", "PP-FormulaNet_plus-S", "PP-FormulaNet_plus-M").contains(modelName)) {
            size = new Size(384, 384);
        } else {
            size = new Size(768, 768);
        }

        this.uniMERNetImgDecode = new UniMERNetImgDecode(size, false);
        this.uniMERNetTestTransform = new UniMERNetTestTransform();
        this.latexImageFormat = new LatexImageFormat();
        this.postProcessor = new UniMERNetDecode(
                new File(modelDir, "tokenizer.json").toPath(),
                Map.of(
                        "maxLength", "4096",
                        "modelMaxLength", "768",
                        "truncation", "longest_first",
                        "stride", "0",
                        "padToMultipleOf", "0"
                )
        );
    }

    @Override
    public List<FormulaRecognitionResult> doBatchPredict(List<PreProcessResult> inputNDArrays, NDManager manager, Map<String, Object> extraParameters) {
        try {
            NDList ndList = new NDList();
            ndList.addAll(PreProcessResult.getNDArrays(inputNDArrays));
            NDArray inputNdArray = NDArrays.stack(ndList, 0);
            inputNDArrays.forEach(IOUtil::close);

            FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            long[] shape = inputNdArray.getShape().getShape();
            OnnxTensor onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
            for (String inputName : inputNames) {
                inputs.put(inputName, onnxInput);
            }
            OrtSession.Result onnxResult = session.run(inputs, outputNames);
            Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
            long[][] preditResult = null;
            List<String> texts = null;
            if (optinalResult.isPresent()) {
                preditResult = (long[][]) optinalResult.get().getValue();
                onnxResult.close();
                onnxInput.close();

                NDArray preds = manager.create(preditResult);
                texts = postProcessor.call(preds);
                preds.close();
            }
            List<FormulaRecognitionResult> results = new ArrayList<>();
            if (preditResult != null && texts != null) {
                for (int i = 0; i < preditResult.length; i++) {
                    long[] tokens = preditResult[i];
                    results.add(new FormulaRecognitionResult(List.of(texts.get(i)), tokens, -1));
                }
            }
            return results;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    @Override
    public PreProcessResult processRgb(Mat rgbMat, NDManager manager) {
        Mat decodeResult = uniMERNetImgDecode.process(rgbMat);
        NDArray uniMERNetImgDecodeResult = ImageUtil.toNDArray(decodeResult, manager, null);
        NDArray uniMERNetTestTransformResult = uniMERNetTestTransform.transform(manager, uniMERNetImgDecodeResult);
        NDArray transformedResult = latexImageFormat.format(uniMERNetTestTransformResult, manager);
        IOUtil.close(uniMERNetTestTransformResult);
        IOUtil.close(uniMERNetImgDecodeResult);
        IOUtil.close(rgbMat);
        return new PreProcessResult(null, transformedResult);
    }

}
