# Model Registry

The Flux OCR project now supports a flexible model registry system that allows models to be registered and looked up dynamically, replacing the previous hardcoded if-else chains.

## Overview

The model registry system consists of two main components:

1. **ModelFactory**: A functional interface for creating model instances
2. **ModelRegistry**: A registry that maps model names to their corresponding factories

## Benefits

- **Extensibility**: New models can be easily registered without modifying existing code
- **Maintainability**: Cleaner code with less duplication and conditional logic
- **Flexibility**: Models can be registered at runtime, enabling plugin architectures
- **Type Safety**: Generic type parameters ensure type-safe model creation

## Core Classes

### ModelFactory

A functional interface for creating model instances:

```java
@FunctionalInterface
public interface ModelFactory<T> {
    T create(String modelRootDir, String modelName, int gpuIndex, OrtEnvironment env);
}
```

### ModelRegistry

A registry for storing and retrieving model factories:

```java
public class ModelRegistry<T> {
    public void register(String modelName, ModelFactory<T> factory);
    public void register(Collection<String> modelNames, ModelFactory<T> factory);
    public boolean isRegistered(String modelName);
    public Optional<ModelFactory<T>> getFactory(String modelName);
    public Set<String> getRegisteredModelNames();
    public boolean unregister(String modelName);
    public void clear();
}
```

## Usage

### Built-in Model Classes

All main model classes (LayoutModel, FormulaRecognitionModel, TextDetectionModel, TextRecognitionModel, TableModel, DocOrientationClassifyModel) now use the registry pattern internally. This is transparent to users - existing code continues to work as before:

```java
// Still works exactly as before
LayoutModel model = new LayoutModel(modelRootDir, modelName, gpuIndex, env);
```

### Accessing the Registry

Each model class exposes its registry via a static `getRegistry()` method:

```java
ModelRegistry<BatchPredictor<ProcessedMat, List<ObjectDetectionResult>>> registry =
    LayoutModel.getRegistry();

// Check if a model is supported
boolean isSupported = registry.isRegistered("docling-layout-egret-large");

// Get all registered model names
Set<String> modelNames = registry.getRegisteredModelNames();
```

### Registering Custom Models

You can register custom models at runtime:

```java
// Get the registry
var registry = LayoutModel.getRegistry();

// Register a custom model
registry.register("my-custom-layout-model", (dir, name, gpu, env) -> {
    return new MyCustomLayoutModel(dir, name, gpu, env);
});

// Now you can use it like any other model
LayoutModel model = new LayoutModel(modelRootDir, "my-custom-layout-model", gpuIndex, env);
```

### Registering Multiple Model Names

If your model implementation supports multiple model names, you can register them all at once:

```java
registry.register(
    List.of("model-v1", "model-v2", "model-v3"),
    MyModelImplementation::new
);
```

## Model Classes

All main model classes have been refactored to use the registry pattern:

### LayoutModel
Supports: Docling and PaddlePaddle layout models

### FormulaRecognitionModel
Supports: Dolphin, Granite, Nougat, Paddle, Pix2Text, TexTeller, Unirec formula models

### TextDetectionModel
Supports: PP-OCRv4 and PP-OCRv5 detection models

### TextRecognitionModel
Supports: PP-OCRv4 and PP-OCRv5 recognition models

### TableModel
Supports: Dolphin and Unirec table models

### DocOrientationClassifyModel
Supports: PaddlePaddle orientation classification models

## Backward Compatibility

The model registry is implemented in a backward-compatible way:

- All existing model names continue to work
- Existing API remains unchanged
- Model construction behavior is identical to previous versions
- Error messages for unsupported models are preserved

## Testing

The model registry includes comprehensive test coverage:

### Unit Tests (ModelRegistryTest)
Tests the core registry functionality including registration, lookup, and removal.

### Integration Tests (ModelRegistryIntegrationTest)
Verifies that all model classes have registered their expected models and maintain backward compatibility.

## Example: Adding a New Model Type

```java
// 1. Create your model class implementing BatchPredictor
public class MyCustomModel extends BatchPredictor<Input, Output> {
    public MyCustomModel(String modelRootDir, String modelName,
                        int gpuIndex, OrtEnvironment env) {
        // Implementation
    }

    // Implement required methods
}

// 2. Register it in your main model class
public class MyModelType extends BatchPredictor<Input, Output> {
    private static final ModelRegistry<BatchPredictor<Input, Output>> REGISTRY =
        new ModelRegistry<>();

    static {
        REGISTRY.register("my-model-name", MyCustomModel::new);
    }

    public MyModelType(String modelRootDir, String modelName,
                      int gpuIndex, OrtEnvironment env) {
        ModelFactory<BatchPredictor<Input, Output>> factory =
            REGISTRY.getFactory(modelName)
                .orElseThrow(() -> new FluxException("Unsupported model: " + modelName));
        this.predictor = factory.create(modelRootDir, modelName, gpuIndex, env);
    }

    public static ModelRegistry<BatchPredictor<Input, Output>> getRegistry() {
        return REGISTRY;
    }
}
```

## Migration Guide

If you had custom model implementations, you can now register them more cleanly:

### Before (manual if-else)
```java
if (modelName.equals("custom-model")) {
    model = new CustomModel(...);
} else if (modelName.equals("another-model")) {
    model = new AnotherModel(...);
}
```

### After (registry pattern)
```java
registry.register("custom-model", CustomModel::new);
registry.register("another-model", AnotherModel::new);

var factory = registry.getFactory(modelName)
    .orElseThrow(() -> new FluxException("Unsupported model: " + modelName));
model = factory.create(...);
```
