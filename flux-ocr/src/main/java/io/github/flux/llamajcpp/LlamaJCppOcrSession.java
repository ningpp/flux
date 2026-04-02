package io.github.flux.llamajcpp;

import io.github.flux.core.TextResult;
import io.github.flux.exception.FluxException;
import io.gravitee.llama.cpp.ConversationState;
import io.gravitee.llama.cpp.DefaultLlamaIterator;
import io.gravitee.llama.cpp.LlamaChatMessage;
import io.gravitee.llama.cpp.LlamaChatMessages;
import io.gravitee.llama.cpp.LlamaContext;
import io.gravitee.llama.cpp.LlamaContextParams;
import io.gravitee.llama.cpp.LlamaModel;
import io.gravitee.llama.cpp.LlamaModelParams;
import io.gravitee.llama.cpp.LlamaOutput;
import io.gravitee.llama.cpp.LlamaSampler;
import io.gravitee.llama.cpp.LlamaTemplate;
import io.gravitee.llama.cpp.LlamaTokenizer;
import io.gravitee.llama.cpp.LlamaVocab;
import io.gravitee.llama.cpp.MtmdContext;
import io.gravitee.llama.cpp.MtmdContextParams;
import io.gravitee.llama.cpp.MtmdImage;
import io.gravitee.llama.cpp.Role;
import org.opencv.core.Mat;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

final class LlamaJCppOcrSession implements AutoCloseable {

    private final LlamaJCppRuntime.Handle runtimeHandle;
    private final Arena arena;
    private final LlamaJCppConfig config;
    private final LlamaModel model;
    private final MtmdContext mtmdContext;
    private final LlamaContext context;
    private final LlamaVocab vocab;
    private final LlamaTemplate template;

    LlamaJCppOcrSession(final LlamaJCppConfig config, final int gpuIndex) {
        this.config = config;
        this.runtimeHandle = LlamaJCppRuntime.acquire();
        Arena localArena = null;
        LlamaModel localModel = null;
        MtmdContext localMtmdContext = null;
        LlamaContext localContext = null;
        LlamaVocab localVocab = null;
        LlamaTemplate localTemplate = null;
        try {
            localArena = Arena.ofShared();
            final LlamaModelParams modelParams = new LlamaModelParams(localArena)
                    .setNThreads(config.nThreads())
                    .nGpuLayers(config.nGpuLayers());
            if (gpuIndex >= 0) {
                modelParams.mainGpu(gpuIndex);
            }
            localModel = new LlamaModel(localArena, config.modelFile(), modelParams);
            final MtmdContextParams mtmdContextParams = new MtmdContextParams(localArena)
                    .useGpu(config.useGpu())
                    .nThreads(config.nThreads())
                    .printTimings(config.printTimings())
                    .mediaMarker(config.mediaMarker());
            localMtmdContext = new MtmdContext(localArena, localModel, config.mmprojFile(), mtmdContextParams);
            localContext = new LlamaContext(localArena, localModel, new LlamaContextParams(localArena)
                    .nCtx(config.contextSize())
                    .nBatch(config.batchSize())
                    .nUBatch(config.batchSize())
                    .nSeqMax(1)
                    .nThreads(config.nThreads())
                    .nThreadsBatch(config.nThreads())
                    .noPerf(!config.printTimings()));
            localVocab = new LlamaVocab(localModel);
            localTemplate = new LlamaTemplate(localModel);
            if (!localMtmdContext.supportsVision()) {
                throw new FluxException("Configured llamaj.cpp model does not support vision input: " + config.modelFile());
            }
            this.arena = localArena;
            this.model = localModel;
            this.mtmdContext = localMtmdContext;
            this.context = localContext;
            this.vocab = localVocab;
            this.template = localTemplate;
        } catch (RuntimeException e) {
            cleanupResources(localContext, localMtmdContext, localModel, localArena);
            runtimeHandle.close();
            throw e;
        } catch (Exception e) {
            cleanupResources(localContext, localMtmdContext, localModel, localArena);
            runtimeHandle.close();
            throw new FluxException("Failed to initialize llamaj.cpp OCR session", e);
        }
    }

    synchronized TextResult predict(final Mat rgbMat, final String prompt) {
        context.clearCache();
        try (Arena requestArena = Arena.ofConfined()) {
            final MtmdImage image = createImage(requestArena, rgbMat);
            final LlamaSampler sampler = createSampler(requestArena);
            try {
                final LlamaTokenizer tokenizer = new LlamaTokenizer(vocab, context);
                final ConversationState state = ConversationState.create(requestArena, context, tokenizer, sampler)
                        .initialize(buildPrompt(requestArena, prompt))
                        .setMedia(List.of(image))
                        .setMaxTokens(config.maxTokens());
                if (!config.stopStrings().isEmpty()) {
                    state.setStopStrings(config.stopStrings());
                }
                final DefaultLlamaIterator iterator = new DefaultLlamaIterator(state, mtmdContext);
                final StringBuilder output = new StringBuilder();
                while (iterator.hasNext()) {
                    final LlamaOutput next = iterator.next();
                    output.append(next.text());
                }
                final String text = output.toString().trim();
                return new TextResult(text, tokenize(tokenizer, requestArena, text), -1f);
            } finally {
                image.free();
                sampler.free();
                context.clearCache();
            }
        }
    }

    private String buildPrompt(final Arena requestArena, final String prompt) {
        final String promptContent = config.buildPromptContent(prompt);
        if (!config.useChatTemplate()) {
            return promptContent;
        }
        final List<LlamaChatMessage> messages = new ArrayList<>();
        if (config.systemPrompt() != null && !config.systemPrompt().isBlank()) {
            messages.add(new LlamaChatMessage(requestArena, Role.SYSTEM, config.systemPrompt()));
        }
        messages.add(new LlamaChatMessage(requestArena, Role.USER, promptContent));
        return template.applyTemplate(requestArena, new LlamaChatMessages(requestArena, messages), config.contextSize());
    }

    private LlamaSampler createSampler(final Arena requestArena) {
        final LlamaSampler sampler = new LlamaSampler(requestArena);
        if (config.temperature() <= 0f) {
            return sampler.greedy();
        }
        return sampler.temperature(config.temperature())
                .topK(config.topK())
                .topP(config.topP(), 1)
                .minP(config.minP(), 1)
                .seed(config.seed());
    }

    private MtmdImage createImage(final Arena requestArena, final Mat rgbMat) {
        if (config.useNativeImageDecoder()) {
            return MtmdImage.fromBytesNative(requestArena, mtmdContext, LlamaJCppImageUtil.toPngBytes(rgbMat));
        }
        return MtmdImage.fromBufferedImage(requestArena, LlamaJCppImageUtil.toBufferedImage(rgbMat));
    }

    private long[] tokenize(final LlamaTokenizer tokenizer, final Arena requestArena, final String text) {
        if (text == null || text.isBlank()) {
            return new long[0];
        }
        final LlamaTokenizer.TokenizerResponse tokens = tokenizer.tokenize(requestArena, text);
        final int[] values = tokens.data()
                .reinterpret((long) tokens.size() * ValueLayout.JAVA_INT.byteSize())
                .toArray(ValueLayout.JAVA_INT);
        final long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    @Override
    public void close() {
        cleanupResources(context, mtmdContext, model, arena);
        runtimeHandle.close();
    }

    private static void cleanupResources(final LlamaContext llamaContext,
                                         final MtmdContext multimodalContext,
                                         final LlamaModel llamaModel,
                                         final Arena closeableArena) {
        closeQuietly(llamaContext);
        closeQuietly(multimodalContext);
        closeQuietly(llamaModel);
        closeQuietly(closeableArena);
    }

    private static void closeQuietly(final Object resource) {
        if (resource == null) {
            return;
        }
        try {
            if (resource instanceof LlamaContext llamaContext) {
                llamaContext.free();
            } else if (resource instanceof MtmdContext multimodalContext) {
                multimodalContext.free();
            } else if (resource instanceof LlamaModel llamaModel) {
                llamaModel.free();
            } else if (resource instanceof Arena closeableArena) {
                closeableArena.close();
            }
        } catch (Exception ignored) {
        }
    }
}
