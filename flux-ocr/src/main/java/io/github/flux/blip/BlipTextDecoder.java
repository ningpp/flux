package io.github.flux.blip;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;

import java.util.Map;

public class BlipTextDecoder implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public BlipTextDecoder(String modelPath, int gpuIndex, OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelPath, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    public float[][][] predict(long[][] inputIds, float[][][] encoderHiddenStates) throws OrtException {
        return predict(inputIds, inputIds[0].length, encoderHiddenStates);
    }

    /**
     * @param seqLen 仅取 {@code inputIds} 每行前 {@code seqLen} 列作为模型输入。
     *               自回归解码时序列逐步增长，复用预分配缓冲、避免每步重复分配与拷贝前缀。
     */
    public float[][][] predict(long[][] inputIds, int seqLen, float[][][] encoderHiddenStates) throws OrtException {
        try (OnnxTensor idsTensor = ArrayUtil.createOnnxTensor(inputIds, seqLen, env);
             OnnxTensor attTensor = ArrayUtil.createOnnxTensor(encoderHiddenStates, env);
             Result result = session.run(Map.of(
                     "input_ids", idsTensor,
                     "encoder_hidden_states", attTensor
             ))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
