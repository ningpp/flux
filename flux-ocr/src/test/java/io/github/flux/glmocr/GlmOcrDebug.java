/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.glmocr;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.Arrays;

/**
 * Debug output for comparing with Python.
 */
public class GlmOcrDebug {

    public static void main(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\datasets\\UniMER-Test\\spe\\0000404.png";
        String modelPath = "D:\\models\\onnx\\GLM-OCR\\vision_encoder.onnx";
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager();
             GlmOcrVisionEncoderModel visionEncoder = new GlmOcrVisionEncoderModel(modelPath, -1, env)) {
            
            // Load image
            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            System.out.println("Image size: " + rgbMat.cols() + "x" + rgbMat.rows());
            
            // Process
            GlmOcrImageProcessor.PreprocessResult ppResult = 
                    GlmOcrImageProcessor.process(rgbMat, matManager, ndManager);
            
            float[][] pixelValues = ppResult.pixelValues;
            int[] gridThw = ppResult.imageGridThw;
            
            System.out.println("Pixel values shape: [" + pixelValues.length + ", " + pixelValues[0].length + "]");
            System.out.println("Grid THW: " + Arrays.toString(gridThw));
            
            // Print first patch stats
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            float sum = 0;
            for (float v : pixelValues[0]) {
                min = Math.min(min, v);
                max = Math.max(max, v);
                sum += v;
            }
            System.out.printf("First patch min/max/mean: %.4f / %.4f / %.4f%n", min, max, sum / pixelValues[0].length);
            
            // Print first few values
            System.out.print("First 10 values of first patch: ");
            for (int i = 0; i < 10; i++) {
                System.out.printf("%.4f ", pixelValues[0][i]);
            }
            System.out.println();
            
            // Print values at key positions to verify layout
            System.out.printf("idx 0 (R_t1 start): %.4f%n", pixelValues[0][0]);
            System.out.printf("idx 196 (R_t2 start): %.4f%n", pixelValues[0][196]);
            System.out.printf("idx 392 (G_t1 start): %.4f%n", pixelValues[0][392]);
            System.out.printf("idx 588 (G_t2 start): %.4f%n", pixelValues[0][588]);
            System.out.printf("idx 784 (B_t1 start): %.4f%n", pixelValues[0][784]);
            System.out.printf("idx 980 (B_t2 start): %.4f%n", pixelValues[0][980]);
            
            // Count unique values in first patch
            int uniqueCount = 0;
            java.util.Set<Float> seen = new java.util.HashSet<>();
            for (float v : pixelValues[0]) {
                if (!seen.contains(v)) {
                    seen.add(v);
                    uniqueCount++;
                }
            }
            System.out.println("First patch unique count: " + uniqueCount);
            
            // Print patch 50 values for comparison (middle patch with content)
            if (pixelValues.length > 50) {
                System.out.println("\nPatch 50, first 10 values:");
                for (int i = 0; i < 10; i++) {
                    System.out.printf("  [%d] = %.10f%n", i, pixelValues[50][i]);
                }
                System.out.println("Patch 50, values at 196, 392, 588:");
                System.out.printf("  [196] = %.10f%n", pixelValues[50][196]);
                System.out.printf("  [392] = %.10f%n", pixelValues[50][392]);
                System.out.printf("  [588] = %.10f%n", pixelValues[50][588]);
            }
            
            // Print pos_ids
            long[][] posIds = GlmOcrVisionEncoderModel.computePosIds(gridThw);
            System.out.println("Pos IDs shape: [" + posIds.length + ", " + posIds[0].length + "]");
            System.out.print("First 5 pos_ids: ");
            for (int i = 0; i < 5; i++) {
                System.out.print(Arrays.toString(posIds[i]) + " ");
            }
            System.out.println();
            System.out.println("posIds[0]: " + Arrays.toString(posIds[0]));
            System.out.println("posIds[1]: " + Arrays.toString(posIds[1]));
            System.out.println("posIds[last]: " + Arrays.toString(posIds[posIds.length - 1]));
            
            // Save inputs as numpy format for comparison
            java.io.DataOutputStream dos = new java.io.DataOutputStream(
                new java.io.FileOutputStream("D:\\tmp\\java_pixel_values.bin"));
            for (float[] row : pixelValues) {
                for (float v : row) {
                    dos.writeFloat(v);
                }
            }
            dos.close();
            
            dos = new java.io.DataOutputStream(
                new java.io.FileOutputStream("D:\\tmp\\java_pos_ids.bin"));
            for (long[] row : posIds) {
                for (long v : row) {
                    dos.writeLong(v);
                }
            }
            dos.close();
            System.out.println("Saved inputs to D:\\tmp\\java_*.bin");
            
            // Run vision encoder
            System.out.println("\n--- Running vision encoder ---");
            float[][] visionOut = visionEncoder.predict(pixelValues, gridThw);
            System.out.println("Vision output shape: [" + visionOut.length + ", " + visionOut[0].length + "]");
            
            // Stats
            float vMin = Float.MAX_VALUE, vMax = Float.MIN_VALUE, vSum = 0;
            for (float[] row : visionOut) {
                for (float v : row) {
                    vMin = Math.min(vMin, v);
                    vMax = Math.max(vMax, v);
                    vSum += v;
                }
            }
            float vMean = vSum / (visionOut.length * visionOut[0].length);
            System.out.printf("Vision output min/max/mean: %.8f / %.8f / %.8f%n", vMin, vMax, vMean);
            
            // First 10 values of first token
            System.out.print("First 10 values of first token: ");
            for (int i = 0; i < 10; i++) {
                System.out.printf("%.8f ", visionOut[0][i]);
            }
            System.out.println();
            
            // Sum of first token
            float tokenSum = 0;
            for (float v : visionOut[0]) {
                tokenSum += v;
            }
            System.out.printf("Sum of first token: %.5f%n", tokenSum);
        }
    }
}
