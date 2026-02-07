package io.github.flux.blip;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;

import java.util.List;
import java.util.Map;

public class BlipDemoTest {

    public static void main(String[] args) throws Exception {
        String modelRootDir = "D:\\models\\onnx";
        String modelName = "blip-image-captioning-large";
        String image1 = "D:\\tmp\\img-2026-02-07-120114.png";
        String image2 = "D:\\tmp\\img-2026-02-07-120018.png";
        
        System.out.println("=".repeat(60));
        System.out.println("BLIP — Java ONNX Inference Demo");
        System.out.println("=".repeat(60));
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            System.out.println("Loading model...");
            try (BlipModel model = new BlipModel(modelRootDir, modelName, -1, env)) {
                System.out.println("Model loaded in " + (System.currentTimeMillis() - t0) + "ms");

                long t1 = System.currentTimeMillis();
                System.out.println("Running inference...");
                List<TextResult> results = model.batchPredictFiles(List.of(image1, image2), 4, matManager, ndManager, Map.of());
                
                long t2 = System.currentTimeMillis();
                System.out.println("Inference time: " + (t2 - t1) + "ms");

                if (!results.isEmpty()) {
                    for (int i = 0; i < results.size(); i++) {
                        System.out.println(String.format("Result %d: ", i+1) + results.get(i).text());
                    }
                } else {
                    System.out.println("No result returned.");
                }
            }
        }
    }
}
