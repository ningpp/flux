package io.github.flux.falconocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.FileReader;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FalconOcrTokenKvDebug {

    static {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    public static void main(String[] args) throws Exception {
        String modelRootDir = args.length > 0 ? args[0] : "D:\\models";
        String modelName = args.length > 1 ? args[1] : "Falcon-OCR-ONNX";
        String imageDir = args.length > 2 ? args[2] : "D:\\models\\falcon-ocr-convert\\imgs";
        String reportPath = args.length > 3 ? args[3] : "D:\\models\\Falcon-OCR-ONNX\\onnx_vs_transformers_e2e.json";
        int gpuIndex = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        String mode = args.length > 5 ? args[5] : "all";

        Map<String, long[]> expected = loadExpectedTokens(reportPath);
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             FalconOcrModel model = FalconOcrModel.getSharedInstance(
                     modelRootDir, modelName, gpuIndex, env, Map.of());
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            if ("single-table1".equals(mode)) {
                verifyCategory(model, matManager, ndManager, imageDir, "table", List.of(
                        "table-2026-01-01-202211.png"
                ), expected);
            } else if ("single-table2".equals(mode)) {
                verifyCategory(model, matManager, ndManager, imageDir, "table", List.of(
                        "table-2026-05-23-124132.png"
                ), expected);
            } else if ("table-only".equals(mode)) {
                verifyCategory(model, matManager, ndManager, imageDir, "table", List.of(
                        "table-2026-01-01-202211.png",
                        "table-2026-05-23-124132.png"
                ), expected);
            } else {
                verifyCategory(model, matManager, ndManager, imageDir, "formula", List.of(
                        "formula-2026-01-18-152316.png",
                        "formula_2025-8-2_17-28-16.jpg"
                ), expected);
                verifyCategory(model, matManager, ndManager, imageDir, "table", List.of(
                        "table-2026-01-01-202211.png",
                        "table-2026-05-23-124132.png"
                ), expected);
            }
        }
    }

    private static void verifyCategory(FalconOcrModel model,
                                       MatManager matManager,
                                       NDManager ndManager,
                                       String imageDir,
                                       String category,
                                       List<String> imageNames,
                                       Map<String, long[]> expected) {
        List<PreProcessResult> inputs = imageNames.stream().map(name -> {
            Mat rgb = ImageUtil.readToRgb(matManager, imageDir + "\\" + name);
            return model.processRgb(matManager, rgb, ndManager);
        }).toList();

        long start = System.currentTimeMillis();
        List<TextResult> results = model.predictCategory(inputs, matManager, category);
        long elapsed = System.currentTimeMillis() - start;

        System.out.printf("%s batch=%d elapsed=%dms%n", category, imageNames.size(), elapsed);
        for (int i = 0; i < imageNames.size(); i++) {
            String name = imageNames.get(i);
            long[] actual = results.get(i).tokens();
            long[] exp = expected.get(name);
            if (exp == null) {
                throw new AssertionError("Missing expected tokens for " + name + " in " + expected.keySet());
            }
            boolean match = Arrays.equals(actual, exp);
            System.out.printf("  %s tokens=%d expected=%d match=%s%n",
                    name, actual.length, exp.length, match);
            if (!match) {
                int mismatch = firstMismatch(actual, exp);
                throw new AssertionError(name + " generated tokens differ. actualFirst="
                        + Arrays.toString(Arrays.copyOf(actual, Math.min(20, actual.length)))
                        + " expectedFirst="
                        + Arrays.toString(Arrays.copyOf(exp, Math.min(20, exp.length)))
                        + " mismatchIndex=" + mismatch
                        + " actualAround=" + around(actual, mismatch)
                        + " expectedAround=" + around(exp, mismatch));
            }
        }
    }

    private static int firstMismatch(long[] actual, long[] expected) {
        int len = Math.min(actual.length, expected.length);
        for (int i = 0; i < len; i++) {
            if (actual[i] != expected[i]) {
                return i;
            }
        }
        return len;
    }

    private static String around(long[] values, int index) {
        int start = Math.max(0, index - 8);
        int end = Math.min(values.length, index + 8);
        return Arrays.toString(Arrays.copyOfRange(values, start, end));
    }

    private static Map<String, long[]> loadExpectedTokens(String reportPath) throws Exception {
        Type type = new TypeToken<Map<String, Object>>() {}.getType();
        Map<String, Object> report;
        try (FileReader reader = new FileReader(Path.of(reportPath).toFile(), StandardCharsets.UTF_8)) {
            report = new Gson().fromJson(reader, type);
        }
        Map<String, long[]> out = new HashMap<>();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> records = (List<Map<String, Object>>) report.get("records");
        for (Map<String, Object> record : records) {
            String category = String.valueOf(record.get("category"));
            if (!"formula".equals(category) && !"table".equals(category)) {
                continue;
            }
            String image = String.valueOf(record.get("image"));
            @SuppressWarnings("unchecked")
            Map<String, Object> transformers = (Map<String, Object>) record.get("transformers");
            @SuppressWarnings("unchecked")
            List<Double> ids = (List<Double>) transformers.get("ids");
            long[] tokens = new long[ids.size()];
            for (int i = 0; i < ids.size(); i++) {
                tokens[i] = ids.get(i).longValue();
            }
            out.put(image, tokens);
        }
        return out;
    }
}
