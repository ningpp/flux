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
import io.github.flux.core.PreProcessResult;
import io.github.flux.core.TextResult;
import io.github.flux.util.ImageUtil;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Tests for GLM-OCR model.
 * 
 * NOTE: These tests require:
 * 1. OpenCV native library loaded (nu.pattern.OpenCV.loadLocally() or System.loadLibrary("opencv"))
 * 2. ONNX model files in the model directory
 * 3. Test images
 * 
 * Set MODEL_ROOT_DIR and TEST_IMAGE_PATH environment variables or modify the constants below.
 */
class GlmOcrModelTest {

    // Update these paths according to your environment
    private static final String MODEL_ROOT_DIR = System.getenv("GLM_OCR_MODEL_DIR") != null
            ? System.getenv("GLM_OCR_MODEL_DIR")
            : "D:/models/onnx";
    
    private static final String TEST_IMAGE_PATH = System.getenv("GLM_OCR_TEST_IMAGE") != null
            ? System.getenv("GLM_OCR_TEST_IMAGE")
            : "D:/test_images/document.png";

    @Test
    @Disabled("Requires ONNX models and test images")
    void testGlmOcrModelInference() throws Exception {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             GlmOcrModel model = new GlmOcrModel(MODEL_ROOT_DIR, "GLM-OCR", -1, env);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            // Load and preprocess image
            Mat rgbMat = ImageUtil.readToRgb(matManager, TEST_IMAGE_PATH);
            PreProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);
            
            // Run batch prediction
            List<TextResult> results = model.doBatchPredict(
                    List.of(ppr),
                    matManager,
                    ndManager,
                    Map.of()
            );
            
            // Verify results
            assertNotNull(results);
            assertFalse(results.isEmpty());
            
            TextResult result = results.get(0);
            assertNotNull(result.getText());
            System.out.println("OCR Result: " + result.getText());
        }
    }

    @Test
    @Disabled("Requires ONNX models and test images")
    void testGlmOcrModelFp16() throws Exception {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             GlmOcrModel model = new GlmOcrModel(MODEL_ROOT_DIR, "GLM-OCR", -1, env, true);  // FP16
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            // Load and preprocess image
            Mat rgbMat = ImageUtil.readToRgb(matManager, TEST_IMAGE_PATH);
            PreProcessResult ppr = model.processRgb(matManager, rgbMat, ndManager);
            
            // Run batch prediction
            List<TextResult> results = model.doBatchPredict(
                    List.of(ppr),
                    matManager,
                    ndManager,
                    Map.of()
            );
            
            // Verify results
            assertNotNull(results);
            assertFalse(results.isEmpty());
            
            TextResult result = results.get(0);
            assertNotNull(result.getText());
            System.out.println("OCR Result (FP16): " + result.getText());
        }
    }

    @Test
    @Disabled("Requires ONNX models and test images")
    void testGlmOcrBatchInference() throws Exception {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             GlmOcrModel model = new GlmOcrModel(MODEL_ROOT_DIR, "GLM-OCR", -1, env);
             MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            // Load same image multiple times for batch test
            Mat rgbMat = ImageUtil.readToRgb(matManager, TEST_IMAGE_PATH);
            PreProcessResult ppr1 = model.processRgb(matManager, rgbMat, ndManager);
            PreProcessResult ppr2 = model.processRgb(matManager, rgbMat.clone(), ndManager);
            
            // Run batch prediction with batch size 2
            List<TextResult> results = model.batchPredict(
                    List.of(ppr1, ppr2),
                    2,  // batch size
                    matManager,
                    ndManager,
                    Map.of()
            );
            
            // Verify results
            assertNotNull(results);
            assertFalse(results.isEmpty());
            
            for (int i = 0; i < results.size(); i++) {
                TextResult result = results.get(i);
                assertNotNull(result.getText());
                System.out.println("OCR Result [" + i + "]: " + result.getText());
            }
        }
    }

    @Test
    @Disabled("Requires ONNX models")
    void testImageProcessorOnly() throws Exception {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
        
        try (MatManager matManager = new MatManager();
             NDManager ndManager = NDManager.newBaseManager()) {
            
            // Load image
            Mat rgbMat = ImageUtil.readToRgb(matManager, TEST_IMAGE_PATH);
            
            // Process image
            var processed = GlmOcrImageProcessor.process(rgbMat, matManager, ndManager);
            
            // Verify output shape [C, H, W]
            long[] shape = processed.getShape().getShape();
            assertNotNull(shape);
            
            System.out.println("Processed image shape: [" + shape[0] + ", " + shape[1] + ", " + shape[2] + "]");
            
            // Verify values are normalized
            float min = processed.min().getFloat();
            float max = processed.max().getFloat();
            System.out.println("Value range: [" + min + ", " + max + "]");
        }
    }
}
