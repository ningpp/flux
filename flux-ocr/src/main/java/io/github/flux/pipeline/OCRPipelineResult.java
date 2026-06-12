package io.github.flux.pipeline;

import com.google.gson.Gson;
import io.github.flux.core.RecognitionResult;

import java.util.List;

public record OCRPipelineResult(int[][] detPolys, List<RecognitionResult> recResults,
                                String docOrientationLabel, float docOrientationScore,
                                String textLineOrientationLabel, float textLineOrientationScore) {

    public OCRPipelineResult(int[][] detPolys, List<RecognitionResult> recResults) {
        this(detPolys, recResults, null, 0f, null, 0f);
    }

    @Override
    public String toString() {
        return "OCRPipelineResult{" +
                "detPolys=" + new Gson().toJson(detPolys) +
                ",\n recResults=" + new Gson().toJson(recResults) +
                ",\n docOrientationLabel=" + docOrientationLabel +
                ", docOrientationScore=" + docOrientationScore +
                ", textLineOrientationLabel=" + textLineOrientationLabel +
                ", textLineOrientationScore=" + textLineOrientationScore +
                "\n}";
    }

}
