package io.github.flux.falconocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxSessionUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class FalconOcrTokenKvModel implements AutoCloseable {

    private static final int NUM_LAYERS = 22;
    private static final int NUM_KV_GROUPS = 2;
    private static final int NUM_HEADS = 16;
    private static final int HEAD_DIM = 64;
    private static final int IMAGE_PATCH_DIM = FalconOcrProcessor.IMAGE_PATCH_DIM;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> requestedOutputNames;
    private final Integer requestedMaxNewTokens;
    private final int maxSeqLen;
    private final OrtIoBindingNative ioBinding;

    FalconOcrTokenKvModel(String modelFile,
                          int gpuIndex,
                          OrtEnvironment env,
                          Integer requestedMaxNewTokens,
                          int maxSeqLen) {
        this.env = env;
        this.requestedMaxNewTokens = requestedMaxNewTokens;
        this.maxSeqLen = maxSeqLen;
        try {
            this.session = OnnxSessionUtil.createSession(env, modelFile, gpuIndex, options -> {
                options.setOptimizationLevel(OptLevel.ALL_OPT);
                options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
            });
            Set<String> modelOutputNames = session.getOutputNames();
            if (!modelOutputNames.contains("next_token")) {
                throw new FluxException("Falcon-OCR requires token-only ONNX output `next_token`: " + modelFile);
            }
            if (!modelOutputNames.contains("present_key_values")) {
                throw new FluxException("Falcon-OCR ONNX missing `present_key_values`: " + modelFile);
            }
            this.requestedOutputNames = Set.of("next_token", "present_key_values");
            this.ioBinding = gpuIndex > -1 ? new OrtIoBindingNative(session, gpuIndex) : null;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    long[][] predict(FalconOcrProcessor.BatchPreprocessed batch) throws OrtException {
        if (ioBinding != null) {
            return predictWithIoBinding(batch);
        }
        return predictWithJavaOrt(batch);
    }

    private long[][] predictWithIoBinding(FalconOcrProcessor.BatchPreprocessed batch) throws OrtException {
        int batchSize = batch.batchSize();
        int generationLimit = generationLimit(batch.paddedPromptLength());
        List<List<Long>> generated = initGenerated(batchSize);
        boolean[] finished = new boolean[batchSize];
        boolean[] activeAttention = initialAttention(batch.batchTokens(), FalconOcrProcessor.PAD_TOKEN_ID);
        long[] currentPosT = maxPositionByBatch(batch.batchPosT());
        int currentLength = batch.paddedPromptLength();
        DecodeBuffers decodeBuffers = new DecodeBuffers(batchSize);

        OnnxTensor tokens = null;
        OnnxTensor imagePatches = null;
        OnnxTensor posT = null;
        OnnxTensor posHw = null;
        OnnxTensor attentionMask = null;
        OnnxTensor emptyPast = null;
        OrtIoBindingNative.BoundResult activeResult = null;
        try {
            tokens = ArrayUtil.createOnnxTensor(batch.batchTokens(), env);
            imagePatches = ArrayUtil.createOnnxTensor(batch.batchImagePatches(), env);
            posT = ArrayUtil.createOnnxTensor(batch.batchPosT(), env);
            posHw = ArrayUtil.createOnnxTensor(batch.batchPosHw(), env);
            attentionMask = ArrayUtil.createOnnxTensor(batch.batchAttentionMask(), env);
            emptyPast = emptyPast(batchSize);
            activeResult = ioBinding.run(tokens, imagePatches, posT, posHw, attentionMask, emptyPast);
        } finally {
            IOUtil.close(tokens);
            IOUtil.close(imagePatches);
            IOUtil.close(posT);
            IOUtil.close(posHw);
            IOUtil.close(attentionMask);
            IOUtil.close(emptyPast);
        }

        try {
            for (int step = 0; step < generationLimit; step++) {
                long[] nextTokens = activeResult.nextTokens();
                boolean[] wasFinished = finished.clone();
                for (int b = 0; b < batchSize; b++) {
                    if (finished[b]) {
                        nextTokens[b] = FalconOcrProcessor.PAD_TOKEN_ID;
                        continue;
                    }
                    generated.get(b).add(nextTokens[b]);
                    if (isStopToken(nextTokens[b])) {
                        finished[b] = true;
                    }
                }
                if (ArrayUtil.allTrue(finished)) {
                    break;
                }
                if (step + 1 >= generationLimit) {
                    break;
                }

                currentLength++;
                for (int b = 0; b < batchSize; b++) {
                    currentPosT[b] += 1L;
                }
                activeAttention = appendAttention(activeAttention, wasFinished);
                decodeBuffers.update(nextTokens, currentPosT, activeAttention, currentLength);

                OnnxTensor decodeTokens = null;
                OnnxTensor decodeImagePatches = null;
                OnnxTensor decodePosT = null;
                OnnxTensor decodePosHw = null;
                OnnxTensor decodeAttentionMask = null;
                OrtIoBindingNative.BoundResult stepResult = null;
                try {
                    decodeTokens = decodeBuffers.tokenTensor(env);
                    decodeImagePatches = decodeBuffers.imagePatchesTensor(env);
                    decodePosT = decodeBuffers.posTTensor(env);
                    decodePosHw = decodeBuffers.posHwTensor(env);
                    decodeAttentionMask = decodeBuffers.attentionMaskTensor(env, currentLength);
                    stepResult = ioBinding.run(
                            decodeTokens,
                            decodeImagePatches,
                            decodePosT,
                            decodePosHw,
                            decodeAttentionMask,
                            activeResult);
                    IOUtil.close(activeResult);
                    activeResult = stepResult;
                    stepResult = null;
                } finally {
                    IOUtil.close(decodeTokens);
                    IOUtil.close(decodeImagePatches);
                    IOUtil.close(decodePosT);
                    IOUtil.close(decodePosHw);
                    IOUtil.close(decodeAttentionMask);
                    IOUtil.close(stepResult);
                }
            }
            return toArrays(generated);
        } finally {
            IOUtil.close(activeResult);
        }
    }

    private long[][] predictWithJavaOrt(FalconOcrProcessor.BatchPreprocessed batch) throws OrtException {
        int batchSize = batch.batchSize();
        int generationLimit = generationLimit(batch.paddedPromptLength());
        List<List<Long>> generated = initGenerated(batchSize);

        Map<String, OnnxTensor> inputs = null;
        Result activeResult = null;
        boolean[] finished = new boolean[batchSize];
        boolean[] activeAttention = initialAttention(batch.batchTokens(), FalconOcrProcessor.PAD_TOKEN_ID);
        long[] currentPosT = maxPositionByBatch(batch.batchPosT());
        int currentLength = batch.paddedPromptLength();
        DecodeBuffers decodeBuffers = new DecodeBuffers(batchSize);

        try {
            inputs = new HashMap<>();
            inputs.put("tokens", ArrayUtil.createOnnxTensor(batch.batchTokens(), env));
            inputs.put("image_patches", ArrayUtil.createOnnxTensor(batch.batchImagePatches(), env));
            inputs.put("pos_t", ArrayUtil.createOnnxTensor(batch.batchPosT(), env));
            inputs.put("pos_hw", ArrayUtil.createOnnxTensor(batch.batchPosHw(), env));
            inputs.put("attention_mask", ArrayUtil.createOnnxTensor(batch.batchAttentionMask(), env));
            inputs.put("past_key_values", emptyPast(batchSize));
            activeResult = session.run(inputs, requestedOutputNames);
        } finally {
            OnnxUtil.closeTensors(inputs);
        }

        try {
            for (int step = 0; step < generationLimit; step++) {
                long[] nextTokens = readNextToken(activeResult, batchSize);
                boolean[] wasFinished = finished.clone();
                for (int b = 0; b < batchSize; b++) {
                    if (finished[b]) {
                        nextTokens[b] = FalconOcrProcessor.PAD_TOKEN_ID;
                        continue;
                    }
                    generated.get(b).add(nextTokens[b]);
                    if (isStopToken(nextTokens[b])) {
                        finished[b] = true;
                    }
                }
                if (ArrayUtil.allTrue(finished)) {
                    break;
                }
                if (step + 1 >= generationLimit) {
                    break;
                }

                currentLength++;
                for (int b = 0; b < batchSize; b++) {
                    currentPosT[b] += 1L;
                }
                activeAttention = appendAttention(activeAttention, wasFinished);

                Map<String, OnnxTensor> decodeInputs = null;
                Result stepResult = null;
                try {
                    decodeBuffers.update(nextTokens, currentPosT, activeAttention, currentLength);
                    decodeInputs = new HashMap<>();
                    decodeInputs.put("tokens", decodeBuffers.tokenTensor(env));
                    decodeInputs.put("image_patches", decodeBuffers.imagePatchesTensor(env));
                    decodeInputs.put("pos_t", decodeBuffers.posTTensor(env));
                    decodeInputs.put("pos_hw", decodeBuffers.posHwTensor(env));
                    decodeInputs.put("attention_mask", decodeBuffers.attentionMaskTensor(env, currentLength));
                    decodeInputs.put("past_key_values", presentKeyValues(activeResult));
                    stepResult = session.run(decodeInputs, requestedOutputNames);

                    IOUtil.close(activeResult);
                    activeResult = stepResult;
                    stepResult = null;
                } finally {
                    if (decodeInputs != null) {
                        decodeInputs.remove("past_key_values");
                    }
                    OnnxUtil.closeTensors(decodeInputs);
                    IOUtil.close(stepResult);
                }
            }
            return toArrays(generated);
        } finally {
            IOUtil.close(activeResult);
        }
    }

    private int generationLimit(int paddedPromptLength) {
        int remaining = maxSeqLen - paddedPromptLength;
        if (remaining <= 0) {
            throw new FluxException("Falcon-OCR prompt length exceeds model context: paddedPromptLength="
                    + paddedPromptLength + ", maxSeqLen=" + maxSeqLen);
        }
        if (requestedMaxNewTokens == null) {
            return remaining;
        }
        return Math.min(requestedMaxNewTokens, remaining);
    }

    private static List<List<Long>> initGenerated(int batchSize) {
        List<List<Long>> generated = new ArrayList<>(batchSize);
        for (int b = 0; b < batchSize; b++) {
            generated.add(new ArrayList<>());
        }
        return generated;
    }

    private OnnxTensor emptyPast(int batchSize) throws OrtException {
        return OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(new float[0]),
                new long[]{NUM_LAYERS, NUM_KV_GROUPS, batchSize, NUM_HEADS, 0, HEAD_DIM}
        );
    }

    private OnnxTensor presentKeyValues(Result result) throws OrtException {
        return (OnnxTensor) result.get("present_key_values").orElseThrow();
    }

    private long[] readNextToken(Result result, int batchSize) throws OrtException {
        Object value = result.get("next_token").orElseThrow().getValue();
        if (value instanceof long[] ids) {
            if (ids.length != batchSize) {
                throw new FluxException("Falcon-OCR next_token batch mismatch: " + ids.length + " != " + batchSize);
            }
            return ids.clone();
        }
        if (value instanceof long[][] ids2d) {
            long[] ids = new long[batchSize];
            for (int b = 0; b < batchSize; b++) {
                ids[b] = ids2d[b][0];
            }
            return ids;
        }
        throw new FluxException("Unsupported Falcon-OCR next_token output: " + value.getClass());
    }

    private static boolean isStopToken(long token) {
        return token == FalconOcrProcessor.EOS_TOKEN_ID
                || token == FalconOcrProcessor.END_OF_QUERY_TOKEN_ID;
    }

    private static boolean[] initialAttention(long[][] tokens, long padTokenId) {
        int batch = tokens.length;
        int seq = tokens[0].length;
        boolean[] out = new boolean[batch * seq];
        int idx = 0;
        for (long[] row : tokens) {
            for (long token : row) {
                out[idx++] = token != padTokenId;
            }
        }
        return out;
    }

    private static boolean[] appendAttention(boolean[] current, boolean[] wasFinished) {
        int batch = wasFinished.length;
        int oldLen = current.length / batch;
        boolean[] out = new boolean[batch * (oldLen + 1)];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(current, b * oldLen, out, b * (oldLen + 1), oldLen);
            out[b * (oldLen + 1) + oldLen] = !wasFinished[b];
        }
        return out;
    }

    private static long[] maxPositionByBatch(long[][] posT) {
        long[] out = new long[posT.length];
        for (int b = 0; b < posT.length; b++) {
            long max = 0L;
            for (long value : posT[b]) {
                if (value > max) {
                    max = value;
                }
            }
            out[b] = max;
        }
        return out;
    }

    private static long[][] toArrays(List<List<Long>> generated) {
        long[][] out = new long[generated.size()][];
        for (int b = 0; b < generated.size(); b++) {
            List<Long> row = generated.get(b);
            out[b] = new long[row.size()];
            for (int i = 0; i < row.size(); i++) {
                out[b][i] = row.get(i);
            }
        }
        return out;
    }

    private static final class DecodeBuffers {
        private final int batchSize;
        private final long[] tokenColumn;
        private final long[] posTColumn;
        private final float[] zeroImagePatches;
        private final float[] nanPosHw;
        private byte[] attentionMask;

        private DecodeBuffers(int batchSize) {
            this.batchSize = batchSize;
            this.tokenColumn = new long[batchSize];
            this.posTColumn = new long[batchSize];
            this.zeroImagePatches = new float[batchSize * IMAGE_PATCH_DIM];
            this.nanPosHw = new float[batchSize * 2];
            for (int b = 0; b < batchSize; b++) {
                nanPosHw[b * 2] = Float.NaN;
                nanPosHw[b * 2 + 1] = Float.NaN;
            }
        }

        private void update(long[] nextTokens,
                            long[] currentPosT,
                            boolean[] activeAttention,
                            int currentLength) {
            System.arraycopy(nextTokens, 0, tokenColumn, 0, batchSize);
            System.arraycopy(currentPosT, 0, posTColumn, 0, batchSize);
            int total = batchSize * currentLength;
            if (attentionMask == null || attentionMask.length != total) {
                attentionMask = new byte[total];
            }
            for (int i = 0; i < total; i++) {
                attentionMask[i] = activeAttention[i] ? (byte) 1 : (byte) 0;
            }
        }

        private OnnxTensor tokenTensor(OrtEnvironment env) throws OrtException {
            return OnnxTensor.createTensor(env, LongBuffer.wrap(tokenColumn), new long[]{batchSize, 1});
        }

        private OnnxTensor posTTensor(OrtEnvironment env) throws OrtException {
            return OnnxTensor.createTensor(env, LongBuffer.wrap(posTColumn), new long[]{batchSize, 1});
        }

        private OnnxTensor imagePatchesTensor(OrtEnvironment env) throws OrtException {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(zeroImagePatches),
                    new long[]{batchSize, 1, IMAGE_PATCH_DIM});
        }

        private OnnxTensor posHwTensor(OrtEnvironment env) throws OrtException {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(nanPosHw), new long[]{batchSize, 1, 2});
        }

        private OnnxTensor attentionMaskTensor(OrtEnvironment env, int currentLength) throws OrtException {
            return OnnxTensor.createTensor(env,
                    java.nio.ByteBuffer.wrap(attentionMask),
                    new long[]{batchSize, 1, currentLength},
                    ai.onnxruntime.OnnxJavaType.BOOL);
        }
    }

    @Override
    public void close() throws Exception {
        IOUtil.close(ioBinding);
        session.close();
    }
}
