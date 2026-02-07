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

/**
 * Key for caching shared model instances based on configuration parameters.
 * Used by models that implement instance sharing to avoid creating multiple
 * expensive instances with the same configuration.
 *
 * @param modelRootDir the root directory containing model files
 * @param modelName the name of the model
 * @param gpuIndex the GPU index to use (-1 for CPU)
 */
public record InstanceKey(String modelRootDir, String modelName, int gpuIndex) {
}
