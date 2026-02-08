# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flux is a Java-based OCR (Optical Character Recognition) framework for processing documents with complex layouts, text, formulas, and tables. The project is in alpha stage and not recommended for production use.

**Technology Stack:**
- Java 25
- Maven (multi-module build with single `flux-ocr` module)
- ONNX Runtime (primary inference engine)
- DJL/PyTorch (alternative inference engine)
- OpenCV (via ByteDeco wrapper)
- JUnit 5 for testing

## Build and Development Commands

```bash
# Build the project
mvn clean compile

# Run all tests
mvn test

# Run a specific test class
mvn test -Dtest=ModelRegistryTest

# Package the project
mvn clean package

# Install to local repository
mvn clean install

# Release build (with GPG signing and source/javadoc jars)
mvn clean deploy -P release
```

## Core Architecture

### Model Registry Pattern

The framework's key architectural pattern replaces hardcoded model selection with a dynamic registry system. This pattern is consistent across all model categories (Layout, Text Detection, Text Recognition, Formula, Table, Doc Orientation).

**Key Components:**

1. **`ModelRegistry<T>`** - Thread-safe registry mapping model names to factories
2. **`ModelFactory<T>`** - Functional interface for creating model instances with 5 parameters: `(modelRootDir, modelName, gpuIndex, OrtEnvironment, customParams)`
3. **`ModelInstanceKey`** - Record for caching shared instances (used by expensive multi-component models)

**Pattern for Adding Models:**

Each concrete model class (e.g., `DoclingLayoutModel`):
1. Defines a `Set<String> MODEL_NAMES` containing supported model names
2. Has a `static` initializer that registers itself with the parent category's registry
3. Constructor matches `ModelFactory` signature

```java
public class DoclingLayoutModel extends BatchPredictor<ProcessedMat, List<ObjectDetectionResult>> {
    public static final Set<String> MODEL_NAMES = Set.of("model-1", "model-2");

    static {
        LayoutModel.getRegistry().register(MODEL_NAMES, DoclingLayoutModel::new);
    }

    public DoclingLayoutModel(String modelRootDir, String modelName, int gpuIndex,
                              OrtEnvironment env, Map<String, Object> customParams) {
        // implementation
    }
}
```

The parent category class (e.g., `LayoutModel`):
1. Has a `private static final ModelRegistry<> REGISTRY`
2. Has a `static` block that loads all concrete model classes via `Class.forName()`
3. Constructor looks up factory from registry and delegates to it
4. Provides `getRegistry()` accessor for registration

### Batch Processing Pattern

All models extend `BatchPredictor<I, O>`:
- `I process(MatManager, String imagePath, NDManager)` - Load and preprocess a single image
- `abstract I processRgb(MatManager, Mat rgbMat, NDManager)` - Preprocess RGB Mat
- `abstract List<O> doBatchPredict(List<I>, MatManager, NDManager, extraParameters)` - Actual inference
- `batchPredict()` and `batchPredictFiles()` handle batching logic automatically

### Instance Sharing Pattern

Expensive models (typically encoder-decoder models like GotOcr2Model, LightOnOcrModel, Qwen3VlModel) implement instance sharing:
- Static `ConcurrentHashMap<ModelInstanceKey, Model>` cache
- Static `getSharedInstance()` factory method
- Reduces memory footprint when same configuration is reused

## Package Structure

- `io.github.flux.core` - Core abstractions (`BatchPredictor`, `Processor`, `ModelRegistry`, `ModelFactory`, result types, `MatManager`)
- `io.github.flux.model` - Model category facades (`LayoutModel`, `TextDetectionModel`, etc.)
- `io.github.flux.docling` - Docling model implementations
- `io.github.flux.paddle` - PaddlePaddle model implementations and processors
- `io.github.flux.dolphin` - Dolphin model implementations
- `io.github.flux.formula.*` - Formula recognition implementations
- `io.github.flux.gotocr2`, `io.github.flux.lightonocr`, `io.github.flux.qwen3vl`, `io.github.flux.blip`, `io.github.flux.llava`, `io.github.flux.glmocr` - Multi-modal OCR models
- `io.github.flux.bytedeco` - OpenCV wrapper (`MatManager`, `OpenCVImage`, `OpenCVImageFactory`)
- `io.github.flux.util` - Utilities (`OnnxUtil`, `ImageUtil`, `ArrayUtil`, `CollectionUtil`, `ParameterUtil`, `IOUtil`)
- `io.github.flux.exception` - `FluxException` wrapper

## Memory Management

- **`MatManager`** - Manages OpenCV `Mat` lifecycle to prevent native memory leaks
- **`NDManager`** (DJL) - Manages NDArray lifecycle
- Always use try-with-resources for `OnnxTensor` and `OrtSession.Result`

## Testing

Tests located in `flux-ocr/src/test/java/`. Key test patterns:
- `ModelRegistryTest` - Registry functionality
- `*ModelTest` - Individual model tests
- `*BatchTest` - Batch processing tests
