package io.github.flux.paddle;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OCRPipelineSingleModelMemoryProbeTest {

    @Test
    void parsePageRangeExpandsInclusiveOneBasedRange() {
        assertEquals(List.of(9, 10, 11, 12), OCRPipelineSingleModelMemoryProbe.parsePageRange("9-12"));
        assertEquals(List.of(1, 2, 3), OCRPipelineMemoryAttributionProbe.parsePageRange("1-3"));
    }

    @Test
    void classifyLayoutLabelsIntoProbeBuckets() {
        assertEquals("formula", OCRPipelineSingleModelMemoryProbe.classifyLabel("display_formula"));
        assertEquals("table", OCRPipelineSingleModelMemoryProbe.classifyLabel("table"));
        assertEquals("text", OCRPipelineSingleModelMemoryProbe.classifyLabel("paragraph_title"));
        assertEquals("image", OCRPipelineSingleModelMemoryProbe.classifyLabel("figure"));
    }

    @Test
    void attributionPercentHandlesZeroTotal() {
        assertEquals(0d, OCRPipelineMemoryAttributionProbe.percent(3d, 0d));
        assertEquals(25d, OCRPipelineMemoryAttributionProbe.percent(2d, 8d));
    }

    @Test
    void requestSettingsKeepDetectionAndLayoutMemorySafeByDefault() {
        OCRPipelineMemoryAttributionProbe.RequestSettings settings =
                OCRPipelineMemoryAttributionProbe.parseRequestSettings(new String[]{
                        "request", "sample.pdf", "1-4", "2", "4", "8", "4"
                });

        assertEquals(4, settings.pageBatchSize());
        assertEquals(8, settings.recognitionBatchSize());
        assertEquals(4, settings.formulaBatchSize());
        assertEquals(1, settings.detectionBatchSize());
        assertEquals(1, settings.layoutBatchSize());
    }

    @Test
    void requestSettingsAllowExplicitDetectionAndLayoutBatches() {
        OCRPipelineMemoryAttributionProbe.RequestSettings settings =
                OCRPipelineMemoryAttributionProbe.parseRequestSettings(new String[]{
                        "request", "sample.pdf", "1-4", "2", "4", "8", "4", "3", "2"
                });

        assertEquals(3, settings.detectionBatchSize());
        assertEquals(2, settings.layoutBatchSize());
    }
}
