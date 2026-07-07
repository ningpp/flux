package io.github.flux.gotocr2;

import ai.djl.ndarray.NDArray;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class GotOcr2EncoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public GotOcr2EncoderModel(final String modelFile,
                               final int gpuIndex,
                               final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * 将预处理的像素 NDArray 直接拼接为 ONNX 输入，避免通过 DJL 默认管理器
     * {@code NDArrays.stack} 分配原生缓冲（该缓冲此前从不释放，导致每次推理
     * 泄漏约 batch*3*1024*1024*4 字节）。直接拼 float[] 既能根治该泄露，
     * 也省去一次 DJL stack 拷贝，更有益于性能。
     */
    public float[][][] predict(List<NDArray> pixel_values) throws OrtException {
        int batch = pixel_values.size();
        long[] shape0 = pixel_values.get(0).getShape().getShape();
        int c = (int) shape0[0];
        int h = (int) shape0[1];
        int w = (int) shape0[2];
        int plane = c * h * w;
        float[] data = new float[batch * plane];
        int offset = 0;
        for (NDArray arr : pixel_values) {
            float[] a = arr.toFloatArray();
            System.arraycopy(a, 0, data, offset, a.length);
            offset += a.length;
        }
        long[] shape = new long[] {batch, c, h, w};
        try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
             Result result = session.run(Map.of("pixel_values", onnxInput))) {
            return (float[][][]) result.get(0).getValue();
        }
    }

}
