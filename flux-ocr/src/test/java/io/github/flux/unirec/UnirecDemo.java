package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TableResult;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import io.github.flux.model.TableModel;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class UnirecDemo {

    static void main_table() throws Exception {
        String root = "D:\\models\\formula";
        String file1 = "D:\\tmp\\table1-1.png";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        int n = 7;
        try (var model = new TableModel(root, "unirec-0.1b", 0, env);) {
            List<TableResult> results = null;
            var start = LocalDateTime.now();
            System.out.println("\n");
            System.out.println("start: \t\t\t" + start);
            for (int i = 0; i < n; i++) {
                try (
                    MatManager matManager = new MatManager();
                    NDManager ndManager = NDManager.newBaseManager()
                ) {
                    results = model.batchPredictFiles(List.of(file1), 1, matManager, ndManager, Map.of());
                    if (i % 10 == 0) {
                        System.out.println("iter: " + String.format("%6d", i) + " ".repeat(7) + LocalDateTime.now());
                    }
                }
            }
            System.out.println("Result: \n" + results.get(0).text());
            var end = LocalDateTime.now();
            double avgCostMs = Duration.between(start, end).toNanos() / 1000_000d / (double) n;
            System.out.println("Avg Cost: " + avgCostMs+ "ms");
            System.out.println("Generated Tokens: " + results.get(0).tokens().length);
            System.out.println(results.get(0).tokens().length / (avgCostMs / 1000.0) + " Tokens/s");
            System.out.println(start + "\t\t" + end);
        }
    }

    static void main() throws Exception {
        String root = "D:\\models\\formula";
        String file1 = "D:\\tmp\\formula-2026-01-18-152316.png";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        int n = 86400 * 3;
        // n = 37;
        Boolean useFastDecode = Boolean.TRUE;
        try (var model = new FormulaRecognitionModel(root, "unirec-0.1b", 0, env);) {
            List<TextResult> results = null;
            for (int i = 0; i < 3; i++) {
                try (
                        MatManager matManager = new MatManager();
                        NDManager ndManager = NDManager.newBaseManager()
                ) {
                    results = model.batchPredictFiles(List.of(file1), 1, matManager, ndManager,
                            Map.of("useFastDecode", useFastDecode));
                }
            }
            var start = LocalDateTime.now();
            System.out.println("\n");
            System.out.println("start: \t\t\t" + start);
            for (int i = 0; i < n; i++) {
                try (
                        MatManager matManager = new MatManager();
                        NDManager ndManager = NDManager.newBaseManager()
                ) {
                    results = model.batchPredictFiles(List.of(file1), 1, matManager, ndManager,
                            Map.of("useFastDecode", useFastDecode));
                    if (i % 10 == 0) {
                        System.out.println("iter: " + String.format("%6d", i) + " ".repeat(7) + LocalDateTime.now()
                            + " ".repeat(11) + "Avg Cost: " + Duration.between(start, LocalDateTime.now()).toNanos() / 1000_000d / (double) (i+1));
                    }
                }
            }
            System.out.println("Result: \n" + results.get(0).text());
            var end = LocalDateTime.now();
            double avgCostMs = Duration.between(start, end).toNanos() / 1000_000d / (double) n;
            System.out.println("Avg Cost: " + avgCostMs+ "ms");
            System.out.println("Generated Tokens: " + results.get(0).tokens().length);
            System.out.println(results.get(0).tokens().length / (avgCostMs / 1000.0) + " Tokens/s");
            System.out.println(start + "\t\t" + end);
        }
    }

}
