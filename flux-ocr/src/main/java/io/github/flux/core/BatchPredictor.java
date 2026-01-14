package io.github.flux.core;

import ai.djl.ndarray.NDManager;
import io.github.flux.util.CollectionUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public abstract class BatchPredictor<I, R> implements AutoCloseable {

    public abstract List<R> doBatchPredict(List<I> mats, MatManager matManager, NDManager manager, Map<String, Object> extraParameters);

    public final I process(MatManager matManager, String image, NDManager manager) {
        return processRgb(matManager, ImageUtil.readToRgb(matManager, image), manager);
    }

    public abstract I processRgb(MatManager matManager, Mat rgbMat, NDManager manager);

    public final List<R> batchPredictFiles(List<String> images, int batchSize, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        if (images == null || images.isEmpty()) {
            return List.of();
        }
        List<R> results = new ArrayList<>();
        List<List<String>> batched = CollectionUtil.split(images, batchSize);
        for (List<String> batch : batched) {
            List<I> preProcessed = batch.stream().map(file -> this.process(matManager, file, manager)).toList();
            results.addAll(batchPredict(preProcessed, batchSize, matManager, manager, extraParameters));
        }
        return results;
    }

    public final List<R> batchPredict(List<I> images, int batchSize, MatManager matManager, NDManager manager, Map<String, Object> extraParameters) {
        if (images == null || images.isEmpty()) {
            return List.of();
        }
        List<R> results = new ArrayList<>();
        List<List<I>> batched = CollectionUtil.split(images, batchSize);
        for (List<I> batch : batched) {
            results.addAll(doBatchPredict(batch, matManager, manager, extraParameters));
        }
        return results;
    }

}
