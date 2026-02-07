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

import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for model factories that can create specific model implementations.
 * This allows for dynamic model registration and lookup without hardcoding model names.
 *
 * @param <T> the type of model produced by registered factories
 */
public class ModelRegistry<T> {

    private final Map<String, ModelFactory<T>> factories = new ConcurrentHashMap<>();

    /**
     * Registers a model factory for the given model name.
     *
     * @param modelName the name of the model
     * @param factory the factory to create instances of this model
     */
    public void register(String modelName, ModelFactory<T> factory) {
        factories.put(modelName, factory);
    }

    /**
     * Registers a model factory for multiple model names.
     *
     * @param modelNames the names of the models
     * @param factory the factory to create instances of these models
     */
    public void register(Collection<String> modelNames, ModelFactory<T> factory) {
        for (String modelName : modelNames) {
            register(modelName, factory);
        }
    }

    /**
     * Checks if a model name is registered.
     *
     * @param modelName the model name to check
     * @return true if the model is registered, false otherwise
     */
    public boolean isRegistered(String modelName) {
        return factories.containsKey(modelName);
    }

    /**
     * Gets the factory for a specific model name.
     *
     * @param modelName the model name
     * @return an Optional containing the factory if found, empty otherwise
     */
    public Optional<ModelFactory<T>> getFactory(String modelName) {
        return Optional.ofNullable(factories.get(modelName));
    }

    /**
     * Gets all registered model names.
     *
     * @return an unmodifiable set of all registered model names
     */
    public Set<String> getRegisteredModelNames() {
        return Set.copyOf(factories.keySet());
    }

    /**
     * Unregisters a model name.
     *
     * @param modelName the model name to unregister
     * @return true if the model was registered and is now removed, false otherwise
     */
    public boolean unregister(String modelName) {
        return factories.remove(modelName) != null;
    }

    /**
     * Clears all registered models.
     */
    public void clear() {
        factories.clear();
    }
}
