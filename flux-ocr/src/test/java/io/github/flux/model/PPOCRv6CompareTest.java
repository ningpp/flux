package io.github.flux.model;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.RecognitionResult;
import io.github.flux.core.TextDetectionResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * PP-OCRv6全变体对比验证测试。
 * 输出与Python相同格式的JSON文件，用于自动化对比。
 *
 * 测试模型: PP-OCRv6_medium, PP-OCRv6_small, PP-OCRv6_tiny
 */
public class PPOCRv6CompareTest {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String IMG_DIR = "D:\\code\\paddle-ocr-model-v6\\imgs";
    private static final String OUTPUT_DIR = "D:\\code\\paddle-ocr-model-v6\\output_java";

    private static final String[][] MODEL_VARIANTS = {
            {"PP-OCRv6_medium_det", "PP-OCRv6_medium_rec", "PP-OCRv6_medium"},
            {"PP-OCRv6_small_det", "PP-OCRv6_small_rec", "PP-OCRv6_small"},
            {"PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec", "PP-OCRv6_tiny"},
    };

    public static void main(String[] args) throws Exception {
        File imgDir = new File(IMG_DIR);
        if (!imgDir.exists()) {
            System.err.println("图片目录不存在: " + IMG_DIR);
            return;
        }

        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        File[] imgFiles = imgDir.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg"));

        if (imgFiles == null || imgFiles.length == 0) {
            System.err.println("未找到测试图片");
            return;
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            for (String[] variant : MODEL_VARIANTS) {
                String detModelName = variant[0];
                String recModelName = variant[1];
                String variantName = variant[2];

                System.out.println("\n=== " + variantName + " ===");

                try (TextDetectionModel detModel = new TextDetectionModel(MODEL_ROOT_DIR, detModelName, env, 0);
                     TextRecognitionModel recModel = new TextRecognitionModel(MODEL_ROOT_DIR, recModelName, env, 0)) {

                    for (File imgFile : imgFiles) {
                        processImage(detModel, recModel, imgFile, outputDir, variantName);
                    }
                } catch (Exception e) {
                    System.err.println("模型 " + variantName + " 加载失败: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        System.out.println("\nDone. JSON files saved to: " + OUTPUT_DIR);
    }

    private static void processImage(TextDetectionModel detModel, TextRecognitionModel recModel,
                                      File imgFile, File outputDir, String variantName) throws Exception {
        String imagePath = imgFile.getAbsolutePath();
        String baseName = imgFile.getName().replaceAll("\\.[^.]+$", "");

        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            int imgW = rgbMat.cols();
            int imgH = rgbMat.rows();

            PreProcessResult detInput = detModel.processRgb(matManager, rgbMat.clone(), ndManager);
            List<TextDetectionResult> detResults = detModel.doBatchPredict(
                    List.of(detInput), matManager, ndManager, null);

            if (detResults.isEmpty()) {
                System.err.println("No text detected: " + imagePath);
                return;
            }

            TextDetectionResult detResult = detResults.get(0);
            int[][][] polys = detResult.polys();

            // Sort by y (top to bottom)
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < polys.length; i++) indices.add(i);
            indices.sort((a, b) -> {
                int yA = polys[a][0][1], yB = polys[b][0][1];
                if (yA != yB) return Integer.compare(yA, yB);
                return Integer.compare(polys[a][0][0], polys[b][0][0]);
            });

            List<int[][]> sortedPolys = new ArrayList<>();
            List<String> recTexts = new ArrayList<>();
            List<Double> recScores = new ArrayList<>();
            List<int[]> recBoxes = new ArrayList<>();

            for (int idx : indices) {
                int[][] poly = polys[idx];

                int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
                int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE;
                for (int[] p : poly) {
                    minX = Math.min(minX, p[0]); minY = Math.min(minY, p[1]);
                    maxX = Math.max(maxX, p[0]); maxY = Math.max(maxY, p[1]);
                }
                recBoxes.add(new int[]{minX, minY, maxX, maxY});

                // Perspective warp crop (matching Python PaddleX)
                Mat cropRgb = ImageUtil.getMinAreaRectCrop(matManager, ndManager, rgbMat, poly);

                PreProcessResult recInput = recModel.processRgb(matManager, cropRgb, ndManager);
                List<List<RecognitionResult>> recResults = recModel.doBatchPredict(
                        List.of(recInput), matManager, ndManager, null);

                sortedPolys.add(poly);
                if (!recResults.isEmpty() && !recResults.get(0).isEmpty()) {
                    RecognitionResult recResult = recResults.get(0).get(0);
                    recTexts.add(recResult.text());
                    recScores.add((double) recResult.scores()[0]);
                } else {
                    recTexts.add("");
                    recScores.add(0.0);
                }
            }

            String json = buildJson(imagePath, sortedPolys, recTexts, recScores, recBoxes);
            String outputPath = new File(outputDir, variantName + "_" + baseName + "_res.json").getAbsolutePath();
            try (FileWriter writer = new FileWriter(outputPath)) {
                writer.write(json);
            }

            System.out.println("  Saved: " + outputPath + " (" + sortedPolys.size() + " regions)");
        }
    }

    private static String buildJson(String inputPath, List<int[][]> polys,
                                     List<String> recTexts, List<Double> recScores,
                                     List<int[]> recBoxes) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("    \"input_path\": ").append(jsonString(inputPath)).append(",\n");
        sb.append("    \"source\": \"java\",\n");

        sb.append("    \"dt_polys\": [\n");
        for (int i = 0; i < polys.size(); i++) {
            int[][] poly = polys.get(i);
            sb.append("        [");
            for (int j = 0; j < poly.length; j++) {
                if (j > 0) sb.append(", ");
                sb.append("[").append(poly[j][0]).append(", ").append(poly[j][1]).append("]");
            }
            sb.append("]");
            if (i < polys.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ],\n");

        sb.append("    \"text_det_params\": {\n");
        sb.append("        \"limit_side_len\": 64,\n");
        sb.append("        \"limit_type\": \"min\",\n");
        sb.append("        \"max_side_limit\": 4000,\n");
        sb.append("        \"thresh\": 0.3,\n");
        sb.append("        \"box_thresh\": 0.6,\n");
        sb.append("        \"unclip_ratio\": 1.5\n");
        sb.append("    },\n");

        sb.append("    \"rec_texts\": [\n");
        for (int i = 0; i < recTexts.size(); i++) {
            sb.append("        ").append(jsonString(recTexts.get(i)));
            if (i < recTexts.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ],\n");

        sb.append("    \"rec_scores\": [\n");
        for (int i = 0; i < recScores.size(); i++) {
            sb.append("        ").append(String.format("%.16f", recScores.get(i)));
            if (i < recScores.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ],\n");

        sb.append("    \"rec_polys\": [\n");
        for (int i = 0; i < polys.size(); i++) {
            int[][] poly = polys.get(i);
            sb.append("        [");
            for (int j = 0; j < poly.length; j++) {
                if (j > 0) sb.append(", ");
                sb.append("[").append(poly[j][0]).append(", ").append(poly[j][1]).append("]");
            }
            sb.append("]");
            if (i < polys.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ],\n");

        sb.append("    \"rec_boxes\": [\n");
        for (int i = 0; i < recBoxes.size(); i++) {
            int[] box = recBoxes.get(i);
            sb.append("        [").append(box[0]).append(", ").append(box[1]).append(", ")
              .append(box[2]).append(", ").append(box[3]).append("]");
            if (i < recBoxes.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("    ]\n");

        sb.append("}\n");
        return sb.toString();
    }

    private static String jsonString(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        sb.append("\"");
        return sb.toString();
    }
}
