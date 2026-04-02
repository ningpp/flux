package io.github.flux.llamajcpp;

import io.github.flux.exception.FluxException;
import io.gravitee.llama.cpp.LlamaRuntime;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;

import java.lang.foreign.Arena;

public final class LlamaJCppRuntime {

    private static final Object LOCK = new Object();
    private static int retainCount = 0;
    private static Arena runtimeArena;

    private LlamaJCppRuntime() {
    }

    public static Handle acquire() {
        synchronized (LOCK) {
            if (retainCount == 0) {
                try {
                    runtimeArena = Arena.ofShared();
                    String libPath = LlamaLibLoader.load();
                    LlamaRuntime.llama_backend_init();
                    LlamaRuntime.ggml_backend_load_all_from_path(runtimeArena, libPath);
                } catch (Exception e) {
                    try {
                        LlamaRuntime.llama_backend_free();
                    } catch (Exception ignored) {
                    }
                    closeRuntimeArena();
                    throw new FluxException("Failed to initialize llamaj.cpp runtime", e);
                }
            }
            retainCount++;
            return new Handle();
        }
    }

    private static void release() {
        synchronized (LOCK) {
            if (retainCount == 0) {
                return;
            }
            retainCount--;
            if (retainCount == 0) {
                try {
                    LlamaRuntime.llama_backend_free();
                } finally {
                    closeRuntimeArena();
                }
            }
        }
    }

    private static void closeRuntimeArena() {
        if (runtimeArena != null) {
            runtimeArena.close();
            runtimeArena = null;
        }
    }

    public static final class Handle implements AutoCloseable {

        private boolean closed;

        private Handle() {
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                LlamaJCppRuntime.release();
            }
        }
    }
}
