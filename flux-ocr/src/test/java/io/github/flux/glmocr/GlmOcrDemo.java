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
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.opencv.core.Mat;

import java.util.Arrays;

/**
 * Demo for GLM-OCR ONNX inference.
 */
public class GlmOcrDemo {

    public static void main1(String[] args) throws Exception {
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\formula_2025-8-2_17-28-16.jpg";
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            GlmOcrImageProcessor.PreprocessResult ppResult =
                    GlmOcrImageProcessor.process(rgbMat, matManager, ndManager);
            System.out.println(ppResult);
        }
    }

    public static void main(String[] args) throws Exception {
        // Model and image paths
        String modelRootDir = "D:\\models\\onnx";
        String imagePath = args.length > 0 ? args[0] : "D:\\tmp\\formula-2026028-105537.jpg";
        boolean useUnified = true;
        boolean useFp16 = false;  // Use FP16 model for reduced memory
        int gpuIndex = 0;       // -1 for CPU, 0 for first GPU
        
        System.out.println("GLM-OCR ONNX Inference Demo");
        System.out.println("===========================");
        System.out.println("Model: " + modelRootDir + "\\GLM-OCR");
        System.out.println("Image: " + imagePath);
        System.out.println("FP16: " + useFp16);
        System.out.println("Device: " + (gpuIndex < 0 ? "CPU" : "GPU " + gpuIndex));
        System.out.println();
        
        long startTime = System.currentTimeMillis();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             GlmOcrModel model = new GlmOcrModel(modelRootDir, "GLM-OCR", gpuIndex, env, useFp16, useUnified);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            long loadTime = System.currentTimeMillis() - startTime;
            System.out.println("Model loaded in " + loadTime + " ms");
            
            // Load image
            Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
            System.out.println("Image size: " + rgbMat.cols() + "x" + rgbMat.rows());
            
            // Run inference
            long inferStart = System.currentTimeMillis();
            TextResult result = model.predict(rgbMat, matManager, ndManager, "Formula Recognition:");
            long inferTime = System.currentTimeMillis() - inferStart;
            
            System.out.println("\nInference time: " + inferTime + " ms");
            System.out.println("Generated " + result.tokens().length + " tokens");
            System.out.println("\n--- OCR Result ---");
            System.out.println(result.text());
            System.out.println("------------------");
            System.out.println(Arrays.toString(result.tokens()));
        }
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("\nTotal time: " + totalTime + " ms");
    }
}
