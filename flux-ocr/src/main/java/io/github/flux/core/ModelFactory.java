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

import java.util.Map;

/**
 * Factory interface for creating model instances.
 *
 * @param <T> the type of model produced by this factory
 */
@FunctionalInterface
public interface ModelFactory<T> {

    /**
     * Creates a model instance with the given parameters.
     *
     * @param modelRootDir the root directory containing model files
     * @param modelName the name of the model
     * @param gpuIndex the GPU index to use (-1 for CPU)
     * @param env the OrtEnvironment for ONNX Runtime
     * @param customParams custom initialization parameters (e.g., encoder GPU, decoder GPU)
     * @return a new model instance
     */
    T create(String modelRootDir, String modelName, int gpuIndex, OrtEnvironment env, Map<String, Object> customParams);
}
