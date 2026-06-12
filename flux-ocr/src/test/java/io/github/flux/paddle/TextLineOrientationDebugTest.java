/*
 * Systematic debug test for textline orientation model.
 * Compares preprocessing steps with Python reference output.
 */
package io.github.flux.paddle;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.core.PreProcessResult;
import io.github.flux.model.TextLineOrientationModel;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ImageUtil;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class TextLineOrientationDebugTest {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String IMG_DIR = "E:\\textline-ori-imgs";
    private static final String DEBUG_DIR = "D:\\code\\flux\\scripts\\pp-ocrv6\\output_textline_ori\\debug";

    private OrtEnvironment env;
    private MatManager matManager;

    @BeforeAll
    void setUp() {
        env = OrtEnvironment.getEnvironment();
        matManager = new MatManager();
    }

    @AfterAll
    void tearDown() throws Exception {
        matManager.close();
    }

    @Test
    void debugPreprocessStepByStep() throws Exception {
        File imgDir = new File(IMG_DIR);
        File[] imgFiles = imgDir.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg"));
        List<File> sortedFiles = new ArrayList<>(List.of(imgFiles));
        sortedFiles.sort((a, b) -> a.getName().compareTo(b.getName()));

        for (File imgFile : sortedFiles) {
            String imgName = imgFile.getName();
            String base = imgName.replace(".png", "");
            System.out.printf("\n========== %s ==========%n", imgName);

            // Step 1: Read RGB
            Mat rgbMat = ImageUtil.readToRgb(matManager, imgFile.getAbsolutePath());
            System.out.printf("  Step1 ReadRGB: rows=%d, cols=%d, channels=%d, type=%d%n",
                    rgbMat.rows(), rgbMat.cols(), rgbMat.channels(), rgbMat.type());

            // Print some pixel values
            double[] pixel00 = rgbMat.get(0, 0);
            double[] pixel01 = rgbMat.get(0, 1);
            System.out.printf("    [0,0]=%s, [0,1]=%s%n",
                    formatPixel(pixel00), formatPixel(pixel01));

            // Step 2: Resize (160, 80)
            ImageProcessor resize = new Resize(160, 80, ai.djl.modality.cv.Image.Interpolation.BILINEAR);
            Mat resized = resize.process(matManager, rgbMat);
            System.out.printf("  Step2 Resize: rows=%d, cols=%d, channels=%d%n",
                    resized.rows(), resized.cols(), resized.channels());
            double[] r00 = resized.get(0, 0);
            double[] r01 = resized.get(0, 1);
            double[] r40_80 = resized.get(Math.min(40, resized.rows()-1), Math.min(80, resized.cols()-1));
            double[] r79_159 = resized.get(79, 159);
            System.out.printf("    [0,0]=%s, [0,1]=%s%n", formatPixel(r00), formatPixel(r01));
            System.out.printf("    [40,80]=%s%n", formatPixel(r40_80));
            System.out.printf("    [79,159]=%s%n", formatPixel(r79_159));

            // Step 3: Normalize
            ImageProcessor normalize = new Normalize(
                    0.00392156862745098,
                    new double[]{0.485, 0.456, 0.406},
                    new double[]{0.229, 0.224, 0.225}
            );
            Mat normed = normalize.process(matManager, resized);
            System.out.printf("  Step3 Normalize: type=%d (should be CV_32F=%d)%n",
                    normed.type(), CvType.CV_32F);
            double[] n00 = normed.get(0, 0);
            double[] n01 = normed.get(0, 1);
            double[] n40_80 = normed.get(40, 80);
            System.out.printf("    [0,0]=%s, [0,1]=%s%n", formatPixel(n00), formatPixel(n01));
            System.out.printf("    [40,80]=%s%n", formatPixel(n40_80));

            // Step 4: ToCHW
            ImageProcessor toChw = new ToCHWImage();
            Mat chw = toChw.process(matManager, normed);
            System.out.printf("  Step4 ToCHW: rows=%d, cols=%d, channels=%d, type=%d%n",
                    chw.rows(), chw.cols(), chw.channels(), chw.type());

            // Print CHW data - read raw float data from the Mat
            int totalFloats = (int) (chw.rows() * chw.cols() * chw.channels());
            float[] chwData = new float[totalFloats];
            chw.get(0, 0, chwData);

            // Print first 20 values
            StringBuilder sb = new StringBuilder("    flat[:20]=[");
            for (int i = 0; i < Math.min(20, chwData.length); i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.6f", chwData[i]));
            }
            sb.append("]");
            System.out.println(sb);

            // For CHW format, C0 starts at 0, C1 at H*W, C2 at 2*H*W
            int hw = 80 * 160;
            System.out.printf("    CHW C0[0:5]=[%s]%n", formatFloats(chwData, 0, 5));
            System.out.printf("    CHW C1[0:5]=[%s]%n", formatFloats(chwData, hw, 5));
            System.out.printf("    CHW C2[0:5]=[%s]%n", formatFloats(chwData, 2*hw, 5));
            // Position [40,80] = 40*160+80 = 6480
            int pos40_80 = 40 * 160 + 80;
            System.out.printf("    CHW C0[40,80]=%.6f%n", chwData[pos40_80]);
            System.out.printf("    CHW C1[40,80]=%.6f%n", chwData[hw + pos40_80]);
            System.out.printf("    CHW C2[40,80]=%.6f%n", chwData[2*hw + pos40_80]);

            // Save the raw float data that will be sent to ONNX
            // This is the same data that matToOnnxTensor reads
            saveDebugData(chwData, Path.of(DEBUG_DIR, base + "_java_chw_raw.bin"));

            // Step 5: ONNX inference with same tensor
            try (TextLineOrientationModel model = new TextLineOrientationModel(
                    MODEL_ROOT_DIR, "PP-LCNet_x1_0_textline_ori", env, -1)) {
                List<String> paths = List.of(imgFile.getAbsolutePath());
                List<ClassificationResult> results = model.batchPredictFiles(
                        paths, 1, matManager, null, new HashMap<>());
                if (!results.isEmpty()) {
                    ClassificationResult r = results.get(0);
                    System.out.printf("  Result: label=%s, score=%.6f%n", r.label(), r.score());
                }
            }
        }
    }

    @Test
    void compareWithPythonTensor() throws Exception {
        // Read Python tensor binary and compare with Java
        File imgDir = new File(IMG_DIR);
        File[] imgFiles = imgDir.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg"));
        List<File> sortedFiles = new ArrayList<>(List.of(imgFiles));
        sortedFiles.sort((a, b) -> a.getName().compareTo(b.getName()));

        System.out.println("\n========== Python vs Java Tensor Comparison ==========");

        for (File imgFile : sortedFiles) {
            String base = imgFile.getName().replace(".png", "");
            Path pythonBin = Path.of(DEBUG_DIR, base + "_input_tensor.bin");
            Path javaBin = Path.of(DEBUG_DIR, base + "_java_chw_raw.bin");

            if (!Files.exists(pythonBin) || !Files.exists(javaBin)) {
                System.out.printf("  %s: Missing files (python=%s, java=%s)%n",
                        base, Files.exists(pythonBin), Files.exists(javaBin));
                continue;
            }

            byte[] pyBytes = Files.readAllBytes(pythonBin);
            byte[] javaBytes = Files.readAllBytes(javaBin);

            // Python tensor: [1, 3, 80, 160] = 38400 floats = 153600 bytes
            // Java raw: [80, 160, 3] Mat data = 38400 floats = 153600 bytes
            int pyFloatCount = pyBytes.length / 4;
            int javaFloatCount = javaBytes.length / 4;
            System.out.printf("  %s: Python=%d floats, Java=%d floats%n", base, pyFloatCount, javaFloatCount);

            float[] pyFloats = new float[pyFloatCount];
            float[] javaFloats = new float[javaFloatCount];
            java.nio.ByteBuffer.wrap(pyBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(pyFloats);
            java.nio.ByteBuffer.wrap(javaBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(javaFloats);

            // Compare - Python is [1,3,80,160] CHW, Java Mat raw is [80,160,3] HWC (after ToCHWImage bug)
            // The Java "CHW" data stored in HWC Mat is the actual data sent to ONNX
            int minLen = Math.min(pyFloatCount, javaFloatCount);
            int diffCount = 0;
            float maxDiff = 0;
            float sumDiff = 0;
            int maxDiffIdx = -1;

            for (int i = 0; i < minLen; i++) {
                float diff = Math.abs(pyFloats[i] - javaFloats[i]);
                if (diff > 0.001f) {
                    diffCount++;
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        maxDiffIdx = i;
                    }
                }
                sumDiff += diff;
            }

            float avgDiff = sumDiff / minLen;
            System.out.printf("    差异>0.001的数量: %d/%d (%.2f%%)%n", diffCount, minLen, 100.0*diffCount/minLen);
            System.out.printf("    最大差异: %.6f @ index %d%n", maxDiff, maxDiffIdx);
            System.out.printf("    平均差异: %.8f%n", avgDiff);

            if (maxDiffIdx >= 0) {
                // Decode position for Python CHW [1,3,80,160]
                int pyC = (maxDiffIdx / (80 * 160)) % 3;
                int pyH = (maxDiffIdx / 160) % 80;
                int pyW = maxDiffIdx % 160;
                System.out.printf("    Python最大差异位置: C=%d,H=%d,W=%d%n", pyC, pyH, pyW);
                System.out.printf("    Python值: %.6f, Java值: %.6f%n", pyFloats[maxDiffIdx], javaFloats[maxDiffIdx]);

                // Print surrounding values for context
                int start = Math.max(0, maxDiffIdx - 3);
                int end = Math.min(minLen, maxDiffIdx + 4);
                System.out.printf("    Python[%d:%d]=[", start, end);
                for (int i = start; i < end; i++) System.out.printf("%.6f ", pyFloats[i]);
                System.out.println("]");
                System.out.printf("    Java[%d:%d]=[", start, end);
                for (int i = start; i < end; i++) System.out.printf("%.6f ", javaFloats[i]);
                System.out.println("]");
            }

            // Check if data layout is CHW or HWC
            // In CHW: first 80*160 values should all be channel 0
            // In HWC: values alternate R,G,B,R,G,B,...
            System.out.printf("    前9个值: Python=[%s]%n", formatFloatArray(pyFloats, 0, 9));
            System.out.printf("    前9个值: Java  =[%s]%n", formatFloatArray(javaFloats, 0, 9));
        }
    }

    @Test
    void testPythonTensorInJavaOnnx() throws Exception {
        // Load Python's preprocessed tensor and run ONNX inference in Java
        // This isolates the ONNX Runtime difference from preprocessing difference
        File imgDir = new File(IMG_DIR);
        File[] imgFiles = imgDir.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg"));
        List<File> sortedFiles = new ArrayList<>(List.of(imgFiles));
        sortedFiles.sort((a, b) -> a.getName().compareTo(b.getName()));

        String modelPath = "D:\\models\\PP-LCNet_x1_0_textline_ori_onnx\\inference.onnx";
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // CPU only
        OrtSession session = env.createSession(modelPath, opts);
        String inputName = List.copyOf(session.getInputNames()).getFirst();

        System.out.println("\n========== Python Tensor -> Java ONNX (CPU) ==========");

        for (File imgFile : sortedFiles) {
            String base = imgFile.getName().replace(".png", "");
            Path pythonBin = Path.of(DEBUG_DIR, base + "_input_tensor.bin");

            if (!Files.exists(pythonBin)) {
                System.out.printf("  %s: Missing Python tensor%n", base);
                continue;
            }

            byte[] pyBytes = Files.readAllBytes(pythonBin);
            float[] pyFloats = new float[pyBytes.length / 4];
            java.nio.ByteBuffer.wrap(pyBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(pyFloats);

            // Create OnnxTensor from Python data [1, 3, 80, 160]
            java.nio.FloatBuffer fb = java.nio.FloatBuffer.wrap(pyFloats);
            long[] shape = new long[]{1, 3, 80, 160};
            try (OnnxTensor onnxInput = OnnxTensor.createTensor(env, fb, shape);
                 OrtSession.Result result = session.run(Map.of(inputName, onnxInput))) {
                float[][] output = (float[][]) result.get(0).getValue();
                System.out.printf("  %s: Java ONNX(CPU) with Python tensor -> [0_degree=%.6f, 180_degree=%.6f]%n",
                        base, output[0][0], output[0][1]);
            }
        }

        session.close();
        opts.close();

        // Now do same with Java preprocessing
        System.out.println("\n========== Java Tensor -> Java ONNX (CPU) ==========");
        try (TextLineOrientationModel model = new TextLineOrientationModel(
                MODEL_ROOT_DIR, "PP-LCNet_x1_0_textline_ori", env, -1)) {
            for (File imgFile : sortedFiles) {
                String imgName = imgFile.getName();
                List<String> paths = List.of(imgFile.getAbsolutePath());
                List<ClassificationResult> results = model.batchPredictFiles(
                        paths, 1, matManager, null, new HashMap<>());
                if (!results.isEmpty()) {
                    ClassificationResult r = results.get(0);
                    System.out.printf("  %s: Java ONNX(CPU) with Java tensor -> label=%s, score=%.6f%n",
                            imgName, r.label(), r.score());
                }
            }
        }
    }

    private String formatPixel(double[] pixel) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < pixel.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.1f", pixel[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private String formatFloats(float[] data, int offset, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count && (offset + i) < data.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.6f", data[offset + i]));
        }
        return sb.toString();
    }

    private String formatFloatArray(float[] data, int offset, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count && (offset + i) < data.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.6f", data[offset + i]));
        }
        return sb.toString();
    }

    private void saveDebugData(float[] data, Path path) throws Exception {
        Files.createDirectories(path.getParent());
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(data.length * 4);
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float f : data) bb.putFloat(f);
        Files.write(path, bb.array());
    }
}
