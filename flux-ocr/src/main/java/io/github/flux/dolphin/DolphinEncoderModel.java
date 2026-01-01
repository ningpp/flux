package io.github.flux.dolphin;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.exception.FluxException;
import io.github.flux.util.IOUtil;
import io.github.flux.util.ImageUtil;
import io.github.flux.util.OnnxUtil;
import org.opencv.core.Mat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class DolphinEncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;
    private final Set<String> outputNames;
    private final OnnxJavaType dtype;

    public DolphinEncoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env,
                               final OnnxJavaType dtype) {
        this.dtype = dtype;
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);

            this.inputNames = session.getInputNames();
            this.outputNames = session.getOutputNames();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] batchPredict(List<Mat> inputMats, NDManager manager) throws OrtException {
        NDList ndList = new NDList();
        for (Mat mat : inputMats) {
            ndList.add(ImageUtil.toChannalNDArrayFloat(mat, manager));
        }
        NDArray inputNdArray = NDArrays.stack(ndList);
        inputMats.forEach(Mat::release);
        long[] shape = inputNdArray.getShape().getShape();
        OnnxTensor onnxInput;
        if (dtype == OnnxJavaType.FLOAT) {
            var dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        } else {
            NDArray expandResultFloat16 = inputNdArray.toType(DataType.FLOAT16, true);
            var buffer = expandResultFloat16.toByteBuffer().asShortBuffer();
            onnxInput = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16);
            IOUtil.close(expandResultFloat16);
        }
        inputNdArray.close();
        IOUtil.close(ndList);

        Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
        for (String inputName : inputNames) {
            inputs.put(inputName, onnxInput);
        }
        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
        float[][][] encodeResultFloats = null;
        if (optinalResult.isPresent()) {
            encodeResultFloats = (float[][][]) optinalResult.get().getValue();
        }
        onnxResult.close();
        OnnxUtil.closeTensors(inputs);
        return encodeResultFloats;
    }

    public float[][][] predict(Mat inputMat, NDManager manager) throws OrtException {
        NDArray inputNdArray = ImageUtil.toChannalNDArrayFloat(inputMat, manager);
        inputMat.release();
        NDArray expandResult = inputNdArray.expandDims(0);
        inputNdArray.close();
        long[] shape = expandResult.getShape().getShape();
        OnnxTensor onnxInput;
        if (dtype == OnnxJavaType.FLOAT) {
            var dataBuffer = expandResult.toByteBuffer().asFloatBuffer();
            onnxInput = OnnxTensor.createTensor(env, dataBuffer, shape);
        } else {
            NDArray expandResultFloat16 = expandResult.toType(DataType.FLOAT16, true);
            var buffer = expandResultFloat16.toByteBuffer().asShortBuffer();
            onnxInput = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16);
            IOUtil.close(expandResultFloat16);
        }
        IOUtil.close(expandResult);

        Map<String, OnnxTensor> inputs = new HashMap<>(inputNames.size());
        for (String inputName : inputNames) {
            inputs.put(inputName, onnxInput);
        }
        OrtSession.Result onnxResult = session.run(inputs, outputNames);
        Optional<OnnxValue> optinalResult = onnxResult.get(List.copyOf(outputNames).get(0));
        float[][][] encodeResultFloats = null;
        if (optinalResult.isPresent()) {
            encodeResultFloats = (float[][][]) optinalResult.get().getValue();
        }
        onnxResult.close();
        OnnxUtil.closeTensors(inputs);
        return encodeResultFloats;
    }

}
