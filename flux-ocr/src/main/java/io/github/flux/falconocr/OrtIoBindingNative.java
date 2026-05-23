package io.github.flux.falconocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import com.sun.jna.Function;
import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.NativeLong;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import io.github.flux.exception.FluxException;

import java.lang.reflect.Field;

final class OrtIoBindingNative implements AutoCloseable {

    private static final int ORT_API_VERSION = 23;
    private static final int ORT_DEVICE_ALLOCATOR = 0;
    private static final int ORT_ARENA_ALLOCATOR = 1;
    private static final int ORT_MEM_TYPE_DEFAULT = 0;

    private static final OrtLibrary ORT = Native.load("onnxruntime", OrtLibrary.class);
    private static final OrtApi API = new OrtApi(ORT.OrtGetApiBase());
    private static final Field SESSION_NATIVE_HANDLE = field(OrtSession.class, "nativeHandle");
    private static final Field TENSOR_NATIVE_HANDLE = field(ai.onnxruntime.OnnxTensorLike.class, "nativeHandle");

    private final Pointer sessionHandle;
    private final Pointer ioBinding;
    private final Pointer cpuMemoryInfo;
    private final Pointer cudaMemoryInfo;
    private final Pointer allocator;

    OrtIoBindingNative(OrtSession session, int gpuIndex) {
        if (gpuIndex < 0) {
            throw new FluxException("Falcon-OCR native I/O binding requires CUDA");
        }
        try {
            this.sessionHandle = pointer(nativeHandle(session, SESSION_NATIVE_HANDLE));
            PointerByReference bindingRef = new PointerByReference();
            API.check(API.call(API.createIoBinding, sessionHandle, bindingRef));
            this.ioBinding = bindingRef.getValue();

            PointerByReference cpuInfoRef = new PointerByReference();
            API.check(API.call(API.createCpuMemoryInfo, ORT_ARENA_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, cpuInfoRef));
            this.cpuMemoryInfo = cpuInfoRef.getValue();

            PointerByReference cudaInfoRef = new PointerByReference();
            API.check(API.call(API.createMemoryInfo, "Cuda", ORT_DEVICE_ALLOCATOR, gpuIndex,
                    ORT_MEM_TYPE_DEFAULT, cudaInfoRef));
            this.cudaMemoryInfo = cudaInfoRef.getValue();

            PointerByReference allocatorRef = new PointerByReference();
            API.check(API.call(API.getAllocatorWithDefaultOptions, allocatorRef));
            this.allocator = allocatorRef.getValue();
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    BoundResult run(OnnxTensor tokens,
                    OnnxTensor imagePatches,
                    OnnxTensor posT,
                    OnnxTensor posHw,
                    OnnxTensor attentionMask,
                    BoundResult pastKeyValues) {
        return run(tokens, imagePatches, posT, posHw, attentionMask, pastKeyValues.ortValue);
    }

    BoundResult run(OnnxTensor tokens,
                    OnnxTensor imagePatches,
                    OnnxTensor posT,
                    OnnxTensor posHw,
                    OnnxTensor attentionMask,
                    OnnxTensor pastKeyValues) {
        return run(tokens, imagePatches, posT, posHw, attentionMask,
                pointer(nativeHandle(pastKeyValues, TENSOR_NATIVE_HANDLE)));
    }

    private BoundResult run(OnnxTensor tokens,
                            OnnxTensor imagePatches,
                            OnnxTensor posT,
                            OnnxTensor posHw,
                            OnnxTensor attentionMask,
                            Pointer pastKeyValues) {
        Pointer outputsArray = null;
        try {
            API.invokeVoid(API.clearBoundInputs, ioBinding);
            API.invokeVoid(API.clearBoundOutputs, ioBinding);
            bindInput("tokens", pointer(nativeHandle(tokens, TENSOR_NATIVE_HANDLE)));
            bindInput("image_patches", pointer(nativeHandle(imagePatches, TENSOR_NATIVE_HANDLE)));
            bindInput("pos_t", pointer(nativeHandle(posT, TENSOR_NATIVE_HANDLE)));
            bindInput("pos_hw", pointer(nativeHandle(posHw, TENSOR_NATIVE_HANDLE)));
            bindInput("attention_mask", pointer(nativeHandle(attentionMask, TENSOR_NATIVE_HANDLE)));
            bindInput("past_key_values", pastKeyValues);
            API.check(API.call(API.bindOutputToDevice, ioBinding, "next_token", cpuMemoryInfo));
            API.check(API.call(API.bindOutputToDevice, ioBinding, "present_key_values", cudaMemoryInfo));
            API.check(API.call(API.runWithBinding, sessionHandle, Pointer.NULL, ioBinding));

            PointerByReference outputsRef = new PointerByReference();
            LongByReference outputCountRef = new LongByReference();
            API.check(API.call(API.getBoundOutputValues, ioBinding, allocator, outputsRef, outputCountRef));
            long outputCount = outputCountRef.getValue();
            if (outputCount != 2L) {
                throw new FluxException("Falcon-OCR native I/O binding expected 2 outputs, got: " + outputCount);
            }
            outputsArray = outputsRef.getValue();
            Pointer nextToken = outputsArray.getPointer(0);
            Pointer presentKeyValues = outputsArray.getPointer(Native.POINTER_SIZE);
            long[] ids = readInt64Tensor(nextToken);
            return new BoundResult(ids, presentKeyValues);
        } catch (Exception e) {
            throw new FluxException(e);
        } finally {
            if (outputsArray != null) {
                API.check(API.call(API.allocatorFree, allocator, outputsArray));
            }
        }
    }

    private long[] readInt64Tensor(Pointer value) {
        PointerByReference dataRef = new PointerByReference();
        API.check(API.call(API.getTensorMutableData, value, dataRef));
        int[] count = tensorShape(value);
        long total = 1L;
        for (int dim : count) {
            total *= dim;
        }
        long[] out = dataRef.getValue().getLongArray(0, Math.toIntExact(total));
        API.invokeVoid(API.releaseValue, value);
        return out;
    }

    private int[] tensorShape(Pointer value) {
        PointerByReference infoRef = new PointerByReference();
        API.check(API.call(API.getTensorTypeAndShape, value, infoRef));
        Pointer info = infoRef.getValue();
        try {
            LongByReference countRef = new LongByReference();
            API.check(API.call(API.getDimensionsCount, info, countRef));
            int rank = Math.toIntExact(countRef.getValue());
            Memory dims = new Memory((long) rank * Long.BYTES);
            API.check(API.call(API.getDimensions, info, dims, new NativeLong(rank)));
            int[] out = new int[rank];
            for (int i = 0; i < rank; i++) {
                out[i] = Math.toIntExact(dims.getLong((long) i * Long.BYTES));
            }
            return out;
        } finally {
            API.invokeVoid(API.releaseTensorTypeAndShapeInfo, info);
        }
    }

    private void bindInput(String name, Pointer value) {
        API.check(API.call(API.bindInput, ioBinding, name, value));
    }

    @Override
    public void close() {
        API.invokeVoid(API.releaseMemoryInfo, cpuMemoryInfo);
        API.invokeVoid(API.releaseMemoryInfo, cudaMemoryInfo);
        API.invokeVoid(API.releaseIoBinding, ioBinding);
    }

    static final class BoundResult implements AutoCloseable {
        private final long[] nextTokens;
        private final Pointer ortValue;

        private BoundResult(long[] nextTokens, Pointer ortValue) {
            this.nextTokens = nextTokens;
            this.ortValue = ortValue;
        }

        long[] nextTokens() {
            return nextTokens.clone();
        }

        @Override
        public void close() {
            API.invokeVoid(API.releaseValue, ortValue);
        }
    }

    private static Field field(Class<?> cls, String name) {
        try {
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private static long nativeHandle(Object target, Field field) {
        try {
            return field.getLong(target);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    private static Pointer pointer(long address) {
        return new Pointer(address);
    }

    private interface OrtLibrary extends Library {
        Pointer OrtGetApiBase();
    }

    private static final class OrtApi {
        private final Function getErrorMessage;
        private final Function releaseStatus;
        private final Function getTensorMutableData;
        private final Function getTensorTypeAndShape;
        private final Function getDimensionsCount;
        private final Function getDimensions;
        private final Function createMemoryInfo;
        private final Function createCpuMemoryInfo;
        private final Function releaseMemoryInfo;
        private final Function allocatorFree;
        private final Function getAllocatorWithDefaultOptions;
        private final Function releaseValue;
        private final Function releaseTensorTypeAndShapeInfo;
        private final Function runWithBinding;
        private final Function createIoBinding;
        private final Function releaseIoBinding;
        private final Function bindInput;
        private final Function bindOutputToDevice;
        private final Function getBoundOutputValues;
        private final Function clearBoundInputs;
        private final Function clearBoundOutputs;

        private OrtApi(Pointer apiBase) {
            Function getApi = Function.getFunction(apiBase.getPointer(0));
            Pointer api = getApi.invokePointer(new Object[]{ORT_API_VERSION});
            this.getErrorMessage = function(api, 2);
            this.getTensorMutableData = function(api, 51);
            this.getDimensionsCount = function(api, 61);
            this.getDimensions = function(api, 62);
            this.getTensorTypeAndShape = function(api, 65);
            this.createMemoryInfo = function(api, 68);
            this.createCpuMemoryInfo = function(api, 69);
            this.allocatorFree = function(api, 76);
            this.getAllocatorWithDefaultOptions = function(api, 78);
            this.releaseStatus = function(api, 93);
            this.releaseMemoryInfo = function(api, 94);
            this.releaseValue = function(api, 96);
            this.releaseTensorTypeAndShapeInfo = function(api, 99);
            this.runWithBinding = function(api, 133);
            this.createIoBinding = function(api, 134);
            this.releaseIoBinding = function(api, 135);
            this.bindInput = function(api, 136);
            this.bindOutputToDevice = function(api, 138);
            this.getBoundOutputValues = function(api, 140);
            this.clearBoundInputs = function(api, 141);
            this.clearBoundOutputs = function(api, 142);
        }

        Pointer call(Function function, Object... args) {
            return function.invokePointer(args);
        }

        void invokeVoid(Function function, Object... args) {
            function.invoke(Void.class, args);
        }

        void check(Pointer status) {
            if (status == null || Pointer.nativeValue(status) == 0L) {
                return;
            }
            String message = getErrorMessage.invokeString(new Object[]{status}, false);
            invokeVoid(releaseStatus, status);
            throw new FluxException(message);
        }

        private static Function function(Pointer api, int index) {
            return Function.getFunction(api.getPointer((long) index * Native.POINTER_SIZE));
        }
    }
}
