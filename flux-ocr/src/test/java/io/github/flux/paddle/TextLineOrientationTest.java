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
package io.github.flux.paddle;

import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.ClassificationResult;
import io.github.flux.core.MatManager;
import io.github.flux.model.TextLineOrientationModel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for PP-LCNet textline orientation classification models.
 * Tests both PP-LCNet_x1_0_textline_ori and PP-LCNet_x0_25_textline_ori.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class TextLineOrientationTest {

    private static final String MODEL_ROOT_DIR = "D:\\models";
    private static final String IMG_DIR = "E:\\textline-ori-imgs";
    private static final String OUTPUT_DIR = "D:\\code\\flux\\scripts\\pp-ocrv6\\output_textline_ori";

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
    void testPPCLNetX10TextlineOri() throws Exception {
        testModel("PP-LCNet_x1_0_textline_ori");
    }

    @Test
    void testPPCLNetX025TextlineOri() throws Exception {
        testModel("PP-LCNet_x0_25_textline_ori");
    }

    private void testModel(String modelName) throws Exception {
        System.out.println("\n========== Testing: " + modelName + " ==========");

        try (TextLineOrientationModel model = new TextLineOrientationModel(MODEL_ROOT_DIR, modelName, env, 0)) {
            // Get test images
            File imgDir = new File(IMG_DIR);
            assertTrue(imgDir.exists() && imgDir.isDirectory(), "Image directory exists: " + IMG_DIR);

            File[] imgFiles = imgDir.listFiles((dir, name) ->
                    name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg") ||
                    name.toLowerCase().endsWith(".jpeg") || name.toLowerCase().endsWith(".bmp"));

            assertNotNull(imgFiles, "Found image files");
            assertTrue(imgFiles.length > 0, "At least one image file exists");

            List<String> imagePaths = new ArrayList<>();
            List<File> sortedFiles = new ArrayList<>();
            for (File f : imgFiles) {
                sortedFiles.add(f);
            }
            sortedFiles.sort((a, b) -> a.getName().compareTo(b.getName()));
            for (File f : sortedFiles) {
                imagePaths.add(f.getAbsolutePath());
            }

            // Run batch prediction
            List<ClassificationResult> results = model.batchPredictFiles(
                    imagePaths, 8, matManager, null, new HashMap<>());

            assertEquals(imagePaths.size(), results.size(), "Result count matches image count");

            // Print and verify results
            StringBuilder jsonBuilder = new StringBuilder();
            jsonBuilder.append("{\n");

            for (int i = 0; i < results.size(); i++) {
                ClassificationResult result = results.get(i);
                String imgName = sortedFiles.get(i).getName();

                System.out.printf("  %-50s -> label=%-12s score=%.6f%n",
                        imgName, result.label(), result.score());

                // Basic assertions
                assertNotNull(result.label(), "Label should not be null");
                assertTrue(result.score() > 0, "Score should be positive");
                assertTrue(result.score() <= 1.0, "Score should be <= 1.0");
                assertTrue(result.label().equals("0_degree") || result.label().equals("180_degree"),
                        "Label should be 0_degree or 180_degree");

                if (i > 0) jsonBuilder.append(",\n");
                jsonBuilder.append(String.format(
                        "  \"%s\": {\"model_name\": \"%s\", \"label\": \"%s\", \"score\": %.6f}",
                        imgName, modelName, result.label(), result.score()));
            }

            jsonBuilder.append("\n}");

            // Save Java results to JSON file
            File outputDir = new File(OUTPUT_DIR);
            outputDir.mkdirs();
            String outputPath = new File(outputDir, modelName + "_java_results.json").getAbsolutePath();
            java.nio.file.Files.writeString(java.nio.file.Path.of(outputPath), jsonBuilder.toString());
            System.out.println("Java results saved to: " + outputPath);
        }
    }

    @Test
    void testModelNames() {
        assertTrue(TextLineOrientationModel.MODEL_NAMES.contains("PP-LCNet_x1_0_textline_ori"));
        assertTrue(TextLineOrientationModel.MODEL_NAMES.contains("PP-LCNet_x0_25_textline_ori"));
    }
}
