package io.github.flux.paddle;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.model.LayoutModel;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.io.FileWriter;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PP-DocLayoutV3 Java (Flux/ONNX) batch verification against Python reference.
 * <p>
 * Runs inference on all images in a directory and saves results to JSON.
 * <pre>
 * Usage (from flux-ocr project root):
 *   mvn exec:java "-Dexec.mainClass=io.github.flux.paddle.PPDocLayoutV3VerifyBatch" "-Dexec.classpathScope=test"
 * </pre>
 */
public class PPDocLayoutV3VerifyBatch {

    static final String MODEL_ROOT_DIR = "D:\\models\\layout";
    static final String MODEL_NAME = "PP-DocLayoutV3";
    static final int BATCH_SIZE = 4;

    public static void main(String[] args) throws Exception {
        String imgDirStr = args.length > 0 ? args[0] : "D:\\data\\DocLayNet-v1.2-imgs";
        String outputStr = args.length > 1 ? args[1] : "D:\\code\\pp-doclayoutv3-convert\\results_java.json";

        int gpuIndex = 0;
        Path imgDir = Paths.get(imgDirStr);
        Path outputPath = Paths.get(outputStr);

        // Collect image files
        List<Path> imgPaths = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(imgDir, "*.{png,jpg,jpeg}")) {
            for (Path p : stream) {
                imgPaths.add(p);
            }
        }
        imgPaths.sort(Path::compareTo);

        if (imgPaths.isEmpty()) {
            System.out.println("错误: 在 " + imgDir + " 下未找到任何图片");
            return;
        }
        System.out.println("共 " + imgPaths.size() + " 张图片");

        Map<String, ImageResult> allResults = new HashMap<>();

        try (var env = OrtEnvironment.getEnvironment();
             var model = new LayoutModel(MODEL_ROOT_DIR, MODEL_NAME, gpuIndex, env);
             var ndManager = NDManager.newBaseManager()) {

            long t0 = System.currentTimeMillis();
            int totalImages = imgPaths.size();

            for (int start = 0; start < totalImages; start += BATCH_SIZE) {
                int end = Math.min(start + BATCH_SIZE, totalImages);
                List<Path> batchPaths = imgPaths.subList(start, end);

                try (var matManager = new MatManager()) {
                    List<ProcessedMat> processedMats = new ArrayList<>();
                    List<Path> validPaths = new ArrayList<>();

                    for (Path p : batchPaths) {
                        try {
                            Mat rgbImg = ImageUtil.readToRgb(matManager, p.toString());
                            ProcessedMat pm = model.processRgb(matManager, rgbImg, ndManager);
                            processedMats.add(pm);
                            validPaths.add(p);
                        } catch (Exception e) {
                            System.out.println("  跳过无法读取的图片 " + p.getFileName() + ": " + e.getMessage());
                        }
                    }

                    if (processedMats.isEmpty()) continue;

                    List<List<ObjectDetectionResult>> batchResults =
                            model.doBatchPredict(processedMats, matManager, ndManager, Map.of());

                    for (int i = 0; i < validPaths.size(); i++) {
                        Path p = validPaths.get(i);
                        List<ObjectDetectionResult> dets = batchResults.get(i);

                        ImageResult ir = new ImageResult();
                        for (ObjectDetectionResult det : dets) {
                            ir.detections.add(new DetResult(
                                    det.label(),
                                    round(det.score(), 6),
                                    new float[]{
                                            round(det.coordinate()[0], 2),
                                            round(det.coordinate()[1], 2),
                                            round(det.coordinate()[2], 2),
                                            round(det.coordinate()[3], 2)
                                    }
                            ));
                        }
                        allResults.put(p.getFileName().toString(), ir);
                    }

                    // Release mats
                    for (ProcessedMat pm : processedMats) {
                        pm.release(matManager);
                    }
                }

                // Progress
                int pct = end * 100 / totalImages;
                System.out.printf("  [%3d%%] %d/%d (%d det)%n", pct, end, totalImages, allResults.size());
            }

            long elapsed = System.currentTimeMillis() - t0;
            System.out.printf("%n推理完成: %d 张图片, 耗时 %.1fs (%.1f img/s)%n",
                    totalImages, elapsed / 1000.0, totalImages * 1000.0 / elapsed);
        }

        // Save JSON
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter writer = new FileWriter(outputPath.toFile())) {
            gson.toJson(allResults, writer);
        }
        System.out.println("结果已保存: " + outputPath);
    }

    private static float round(float value, int places) {
        double scale = Math.pow(10, places);
        return (float) (Math.round(value * scale) / scale);
    }

    // Gson-serializable result classes
    static class ImageResult {
        @SuppressWarnings("unused")
        List<DetResult> detections = new ArrayList<>();
    }

    record DetResult(String label, float score, float[] box) {}
}
