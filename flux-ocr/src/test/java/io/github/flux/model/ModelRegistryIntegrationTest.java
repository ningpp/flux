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
package io.github.flux.model;

import io.github.flux.exception.FluxException;
import io.github.flux.llamajcpp.LlamaJCppOcrModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ModelRegistryIntegrationTest {

    @Test
    void testLayoutModelRegistryHasAllExpectedModels() {
        var registry = LayoutModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : LayoutModel.MODEL_NAMES) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in LayoutModel registry");
        }
    }

    @Test
    void testFormulaRecognitionModelRegistryHasAllExpectedModels() {
        var registry = FormulaRecognitionModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : FormulaRecognitionModel.MODEL_NAMES) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in FormulaRecognitionModel registry");
        }
    }

    @Test
    void testLlamaJCppOcrModelIsRegisteredThroughFormulaRegistry() {
        var registry = FormulaRecognitionModel.getRegistry();
        for (String modelName : LlamaJCppOcrModel.MODEL_NAMES) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in FormulaRecognitionModel registry");
        }
    }

    @Test
    void testTextDetectionModelRegistryHasAllExpectedModels() {
        var registry = TextDetectionModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : TextDetectionModel.SUPPORT_MODELS) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in TextDetectionModel registry");
        }
    }

    @Test
    void testTextRecognitionModelRegistryHasAllExpectedModels() {
        var registry = TextRecognitionModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : TextRecognitionModel.SUPPORT_MODELS) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in TextRecognitionModel registry");
        }
    }

    @Test
    void testTableModelRegistryHasAllExpectedModels() {
        var registry = TableModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : TableModel.MODEL_NAMES) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in TableModel registry");
        }
    }

    @Test
    void testDocOrientationClassifyModelRegistryHasAllExpectedModels() {
        var registry = DocOrientationClassifyModel.getRegistry();
        assertNotNull(registry);

        for (String modelName : DocOrientationClassifyModel.MODEL_NAMES) {
            assertTrue(registry.isRegistered(modelName),
                    "Model " + modelName + " should be registered in DocOrientationClassifyModel registry");
        }
    }

    @Test
    void testLayoutModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new LayoutModel("", "UnsupportedModel", -1, null);
        });
    }

    @Test
    void testFormulaRecognitionModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new FormulaRecognitionModel("", "UnsupportedModel", -1, null);
        });
    }

    @Test
    void testTextDetectionModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new TextDetectionModel("", "UnsupportedModel", null, -1);
        });
    }

    @Test
    void testTextRecognitionModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new TextRecognitionModel("", "UnsupportedModel", null, -1);
        });
    }

    @Test
    void testTableModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new TableModel("", "UnsupportedModel", -1, null);
        });
    }

    @Test
    void testDocOrientationClassifyModelThrowsExceptionForUnsupportedModel() {
        assertThrows(FluxException.class, () -> {
            new DocOrientationClassifyModel("", "UnsupportedModel", null, -1);
        });
    }
}
