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
package io.github.flux.core;

import ai.onnxruntime.OrtEnvironment;
import io.github.flux.exception.FluxException;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ModelRegistryTest {

    @Test
    void testRegisterAndLookup() {
        ModelRegistry<String> registry = new ModelRegistry<>();
        ModelFactory<String> factory = (dir, name, gpu, env) -> "TestModel-" + name;

        registry.register("model1", factory);

        assertTrue(registry.isRegistered("model1"));
        assertFalse(registry.isRegistered("model2"));

        var result = registry.getFactory("model1");
        assertTrue(result.isPresent());
        assertEquals("TestModel-model1", result.get().create("dir", "model1", -1, null));
    }

    @Test
    void testRegisterMultipleNames() {
        ModelRegistry<String> registry = new ModelRegistry<>();
        ModelFactory<String> factory = (dir, name, gpu, env) -> "Model-" + name;

        registry.register(List.of("model1", "model2", "model3"), factory);

        assertTrue(registry.isRegistered("model1"));
        assertTrue(registry.isRegistered("model2"));
        assertTrue(registry.isRegistered("model3"));
        assertFalse(registry.isRegistered("model4"));
    }

    @Test
    void testGetRegisteredModelNames() {
        ModelRegistry<String> registry = new ModelRegistry<>();
        ModelFactory<String> factory = (dir, name, gpu, env) -> "Model";

        registry.register("model1", factory);
        registry.register("model2", factory);
        registry.register("model3", factory);

        Set<String> names = registry.getRegisteredModelNames();
        assertEquals(3, names.size());
        assertTrue(names.contains("model1"));
        assertTrue(names.contains("model2"));
        assertTrue(names.contains("model3"));
    }

    @Test
    void testUnregister() {
        ModelRegistry<String> registry = new ModelRegistry<>();
        ModelFactory<String> factory = (dir, name, gpu, env) -> "Model";

        registry.register("model1", factory);
        assertTrue(registry.isRegistered("model1"));

        boolean removed = registry.unregister("model1");
        assertTrue(removed);
        assertFalse(registry.isRegistered("model1"));

        boolean removedAgain = registry.unregister("model1");
        assertFalse(removedAgain);
    }

    @Test
    void testClear() {
        ModelRegistry<String> registry = new ModelRegistry<>();
        ModelFactory<String> factory = (dir, name, gpu, env) -> "Model";

        registry.register("model1", factory);
        registry.register("model2", factory);
        assertEquals(2, registry.getRegisteredModelNames().size());

        registry.clear();
        assertEquals(0, registry.getRegisteredModelNames().size());
        assertFalse(registry.isRegistered("model1"));
        assertFalse(registry.isRegistered("model2"));
    }

    @Test
    void testGetFactoryForUnregisteredModel() {
        ModelRegistry<String> registry = new ModelRegistry<>();

        var result = registry.getFactory("nonexistent");
        assertFalse(result.isPresent());
    }

    @Test
    void testFactoryCanCreateDifferentModels() {
        ModelRegistry<String> registry = new ModelRegistry<>();

        ModelFactory<String> factory1 = (dir, name, gpu, env) -> "Factory1-" + name;
        ModelFactory<String> factory2 = (dir, name, gpu, env) -> "Factory2-" + name;

        registry.register("model1", factory1);
        registry.register("model2", factory2);

        String result1 = registry.getFactory("model1").get().create("", "model1", -1, null);
        String result2 = registry.getFactory("model2").get().create("", "model2", -1, null);

        assertEquals("Factory1-model1", result1);
        assertEquals("Factory2-model2", result2);
    }
}
