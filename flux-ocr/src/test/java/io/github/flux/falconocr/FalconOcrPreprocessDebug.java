package io.github.flux.falconocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDManager;
import io.github.flux.core.MatManager;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.nio.file.Paths;
import java.util.List;

public class FalconOcrPreprocessDebug {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    private record Case(String image, String category, int expectedPromptLength) {
    }

    public static void main(String[] args) throws Exception {
        String modelDir = args.length > 0 ? args[0] : "D:\\models\\Falcon-OCR-ONNX";
        String imageDir = args.length > 1 ? args[1] : "D:\\models\\falcon-ocr-convert\\imgs";
        List<Case> cases = List.of(
                new Case("formula-2026-01-18-152316.png", "formula", 400),
                new Case("formula_2025-8-2_17-28-16.jpg", "formula", 280),
                new Case("table-2026-01-01-202211.png", "table", 3472),
                new Case("table-2026-05-23-124132.png", "table", 1168)
        );

        try (MatManager matManager = new MatManager();
             NDManager ignored = NDManager.newBaseManager();
             HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(modelDir))) {
            List<FalconOcrProcessor.Preprocessed> formulaItems = new java.util.ArrayList<>();
            List<FalconOcrProcessor.Preprocessed> tableItems = new java.util.ArrayList<>();

            for (Case c : cases) {
                Mat rgb = ImageUtil.readToRgb(matManager, imageDir + "\\" + c.image);
                FalconOcrProcessor.Preprocessed item = FalconOcrProcessor.process(
                        matManager, rgb, tokenizer, c.category);
                System.out.printf(
                        "%s category=%s size=%dx%d patches=%dx%d tokens=%d%n",
                        c.image,
                        c.category,
                        item.width(),
                        item.height(),
                        item.patchRows(),
                        item.patchCols(),
                        item.tokens().length
                );
                if (item.tokens().length != c.expectedPromptLength) {
                    throw new AssertionError(c.image + " prompt length expected "
                            + c.expectedPromptLength + " but got " + item.tokens().length);
                }
                if ("formula".equals(c.category)) {
                    formulaItems.add(item);
                } else {
                    tableItems.add(item);
                }
            }

            assertBatch(FalconOcrProcessor.batchPad(formulaItems), 2, 400);
            assertBatch(FalconOcrProcessor.batchPad(tableItems), 2, 3472);
            System.out.println("Falcon-OCR preprocessing matches expected Python prompt lengths.");
        }
    }

    private static void assertBatch(FalconOcrProcessor.BatchPreprocessed batch,
                                    int expectedBatch,
                                    int expectedPaddedPromptLength) {
        if (batch.batchSize() != expectedBatch) {
            throw new AssertionError("batch size expected " + expectedBatch + " but got " + batch.batchSize());
        }
        if (batch.paddedPromptLength() != expectedPaddedPromptLength) {
            throw new AssertionError("padded prompt length expected "
                    + expectedPaddedPromptLength + " but got " + batch.paddedPromptLength());
        }
        System.out.printf(
                "batch=%d paddedPrompt=%d imagePatchDim=%d%n",
                batch.batchSize(),
                batch.paddedPromptLength(),
                batch.batchImagePatches()[0][0].length
        );
    }
}
