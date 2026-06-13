package io.github.flux.pipeline;

import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;

import java.util.List;

/**
 * Represents a single layout region detected by layout analysis,
 * along with its recognition result based on the region type.
 * <p>
 * Region types:
 * <ul>
 *   <li>text — OCR text lines (textResults)</li>
 *   <li>formula — LaTeX formula (formulaResult)</li>
 *   <li>table — HTML table (tableResult)</li>
 *   <li>image — no recognition, just the region info</li>
 * </ul>
 */
public record LayoutRegionResult(
        ObjectDetectionResult layoutRegion,
        String regionType,
        // text region: OCR results
        List<OCRPipelineResult> textResults,
        // formula region: formula recognition result
        TextResult formulaResult,
        // table region: table recognition result
        TableResult tableResult
) {

    public static LayoutRegionResult text(ObjectDetectionResult region, List<OCRPipelineResult> textResults) {
        return new LayoutRegionResult(region, "text", textResults, null, null);
    }

    public static LayoutRegionResult formula(ObjectDetectionResult region, TextResult formulaResult) {
        return new LayoutRegionResult(region, "formula", null, formulaResult, null);
    }

    public static LayoutRegionResult table(ObjectDetectionResult region, TableResult tableResult) {
        return new LayoutRegionResult(region, "table", null, null, tableResult);
    }

    public static LayoutRegionResult image(ObjectDetectionResult region) {
        return new LayoutRegionResult(region, "image", null, null, null);
    }

    /**
     * Get the recognized text content for this region.
     * For text regions, concatenates all recognition results.
     * For formula regions, returns the LaTeX text.
     * For table regions, returns the HTML text.
     * For image regions, returns empty string.
     */
    public String getText() {
        return switch (regionType) {
            case "text" -> textResults == null ? "" :
                    textResults.stream()
                            .flatMap(r -> r.recResults().stream())
                            .map(RecognitionResult::text)
                            .reduce("", (a, b) -> a + b);
            case "formula" -> formulaResult == null ? "" : formulaResult.text();
            case "table" -> tableResult == null ? "" : tableResult.text();
            default -> "";
        };
    }

    @Override
    public String toString() {
        String content = getText();
        if (content.length() > 200) {
            content = content.substring(0, 200) + "...";
        }
        return "LayoutRegionResult{" +
                "label=" + layoutRegion.label() +
                ", score=" + layoutRegion.score() +
                ", regionType=" + regionType +
                ", content=" + content +
                '}';
    }
}
