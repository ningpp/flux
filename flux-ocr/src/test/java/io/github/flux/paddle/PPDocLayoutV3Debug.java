package io.github.flux.paddle;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.MatManager;
import io.github.flux.core.ObjectDetectionResult;
import io.github.flux.core.ProcessedMat;
import io.github.flux.paddle.processor.ImageProcessor;
import io.github.flux.paddle.processor.Normalize;
import io.github.flux.paddle.processor.PPDocLayoutV3PostProcessor;
import io.github.flux.paddle.processor.Resize;
import io.github.flux.paddle.processor.ToCHWImage;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Debug version that dumps order sequence computation details.
 */
public class PPDocLayoutV3Debug {

    static final String MODEL_ROOT_DIR = "D:\\models\\layout";
    static final String MODEL_NAME = "PP-DocLayoutV3";
    static final List<String> LABEL_LIST = List.of(
            "abstract", "algorithm", "aside_text", "chart", "content",
            "display_formula", "doc_title", "figure_title", "footer",
            "footer_image", "footnote", "formula_number", "header",
            "header_image", "image", "inline_formula", "number",
            "paragraph_title", "reference", "reference_content", "seal",
            "table", "text", "vertical_text", "vision_footnote"
    );

    public static void main(String[] args) throws Exception {
        try (var env = OrtEnvironment.getEnvironment();
             var ndManager = NDManager.newBaseManager();
             var matManager = new MatManager()) {

            // Setup session
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            String modelFile = new File(MODEL_ROOT_DIR + File.separator + MODEL_NAME, "model.onnx").getAbsolutePath();
            OrtSession session = env.createSession(modelFile, options);
            String inputName = List.copyOf(session.getInputNames()).getFirst();

            // Preprocess
            String imgPath = "d:\\code\\pp-doclayoutv3-convert\\imgs\\deepseek-v4-26-6-7-1132.png";
            Mat rgbImg = ImageUtil.readToRgb(matManager, imgPath);
            int origW = rgbImg.width(), origH = rgbImg.height();
            System.out.printf("Original size: %d x %d%n", origW, origH);

            List<ImageProcessor> preprocessors = List.of(
                    new Resize(800, 800, Imgproc.INTER_CUBIC),
                    new Normalize(1.0 / 255.0, new double[]{0, 0, 0}, new double[]{1, 1, 1}),
                    new ToCHWImage()
            );

            List<Mat> processed = List.of(rgbImg);
            for (ImageProcessor p : preprocessors) {
                processed = p.process(matManager, processed);
            }

            // Run ONNX
            var onnxInput = ImageUtil.matToOnnxTensor(processed, env);
            var onnxResult = session.run(Map.of(inputName, onnxInput));

            // Get outputs
            float[][][] logitsArr = (float[][][]) onnxResult.get("logits").get().getValue();
            float[][][] predBoxesArr = (float[][][]) onnxResult.get("pred_boxes").get().getValue();
            float[][][] orderLogitsArr = (float[][][]) onnxResult.get("order_logits").get().getValue();

            System.out.printf("logits shape: [%d, %d, %d]%n", logitsArr.length, logitsArr[0].length, logitsArr[0][0].length);
            System.out.printf("pred_boxes shape: [%d, %d, %d]%n", predBoxesArr.length, predBoxesArr[0].length, predBoxesArr[0][0].length);
            System.out.printf("order_logits shape: [%d, %d, %d]%n", orderLogitsArr.length, orderLogitsArr[0].length, orderLogitsArr[0].length);

            // Compute order_seqs manually
            int seqLen = orderLogitsArr[0].length;
            float[][] orderScores = new float[seqLen][seqLen];
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    orderScores[i][j] = (float)(1.0 / (1.0 + Math.exp(-orderLogitsArr[0][i][j])));
                }
            }

            // Compute order_votes (column-wise, matching PyTorch sum(dim=1))
            float[] orderVotes = new float[seqLen];
            for (int j = 0; j < seqLen; j++) {
                float triuSum = 0;
                float trilSum = 0;
                for (int i = 0; i < seqLen; i++) {
                    if (i < j) triuSum += orderScores[i][j];
                    if (i > j) trilSum += (1.0f - orderScores[j][i]);
                }
                orderVotes[j] = triuSum + trilSum;
            }

            // Print first 20 order_votes
            System.out.print("\nJava order_votes[0:20]: [");
            for (int i = 0; i < 20; i++) {
                if (i > 0) System.out.print(", ");
                System.out.printf("%.4f", orderVotes[i]);
            }
            System.out.println("]");

            // Argsort
            Integer[] pointers = new Integer[seqLen];
            for (int i = 0; i < seqLen; i++) pointers[i] = i;
            java.util.Arrays.sort(pointers, java.util.Comparator.comparingDouble(i -> orderVotes[i]));

            int[] orderSeq = new int[seqLen];
            for (int k = 0; k < seqLen; k++) {
                orderSeq[pointers[k]] = k;
            }

            // Print order_seq for detected queries
            System.out.println("\nJava order_seqs for detected queries:");
            int[] detectedQueries = {0, 1, 2, 3, 4, 5, 6, 7, 9, 286};
            for (int q : detectedQueries) {
                System.out.printf("  query %3d -> order_seq=%d%n", q, orderSeq[q]);
            }

            // Python reference order_seqs:
            // query 0 -> 172, query 1 -> 228, query 2 -> 266, query 3 -> 1,
            // query 4 -> 80, query 5 -> 63, query 6 -> 274, query 7 -> 67,
            // query 9 -> 39, query 286 -> 52
            System.out.println("\nPython torch order_seqs for detected queries:");
            System.out.println("  query   0 -> order_seq=172");
            System.out.println("  query   1 -> order_seq=228");
            System.out.println("  query   2 -> order_seq=266");
            System.out.println("  query   3 -> order_seq=1");
            System.out.println("  query   4 -> order_seq=80");
            System.out.println("  query   5 -> order_seq=63");
            System.out.println("  query   6 -> order_seq=274");
            System.out.println("  query   7 -> order_seq=67");
            System.out.println("  query   9 -> order_seq=39");
            System.out.println("  query 286 -> order_seq=52");

            onnxInput.close();
            onnxResult.close();
            session.close();
            options.close();
        }
    }
}
