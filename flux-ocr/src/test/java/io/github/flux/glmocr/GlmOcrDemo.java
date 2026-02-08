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
        boolean useUnified = true;
        boolean useFp16 = false;
        int gpuIndex = 0;

        String[] images = {
            "D:\\datasets\\UniMER-Test\\spe\\0000404.png",
            "D:\\datasets\\UniMER-Test\\spe\\0000317.png",
            "D:\\datasets\\UniMER-Test\\spe\\0000357.png",
        };
        
        long startTime = System.currentTimeMillis();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             GlmOcrModel model = new GlmOcrModel(modelRootDir, "GLM-OCR", gpuIndex, env, useFp16, useUnified);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            long loadTime = System.currentTimeMillis() - startTime;
            System.out.println("Model loaded in " + loadTime + " ms");

            for (String imagePath : images) {
                System.out.println("\n============================================================");
                System.out.println("Image: " + imagePath);
                
                Mat rgbMat = ImageUtil.readToRgb(matManager, imagePath);
                System.out.println("Image size: " + rgbMat.cols() + "x" + rgbMat.rows());
                
                long inferStart = System.currentTimeMillis();
                TextResult result = model.predict(rgbMat, matManager, ndManager, "Formula Recognition:");
                long inferTime = System.currentTimeMillis() - inferStart;
                
                System.out.println("Inference time: " + inferTime + " ms");
                System.out.println("Generated " + result.tokens().length + " tokens");
                System.out.println("Tokens: " + Arrays.toString(result.tokens()));
                System.out.println("Text: " + result.text());
            }
        }
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("\nTotal time: " + totalTime + " ms");
    }
}
