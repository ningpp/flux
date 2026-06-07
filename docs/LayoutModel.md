
## Example
```java

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.model.LayoutModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class LayoutDemo {

  static void main() throws Exception {
    var rootDir = "D:\\models\\layout";
    var img = "d:\\temp\\deepseek-v4-26-6-7-1132.png";
    var modelNames = new ArrayList<>(LayoutModel.MODEL_NAMES)
        .stream().sorted().toList();
    for (var modelName : modelNames) {
      try (var env = OrtEnvironment.getEnvironment();
           var model = new LayoutModel(rootDir, modelName, 0, env);
           var ndManager = NDManager.newBaseManager();
           var matManager = new MatManager()) {
        System.out.println("=".repeat(11) + modelName + "=".repeat(11));
        var allResults = model.batchPredictFiles(List.of(img), 1, matManager, ndManager, Map.of());
        for (var fileResults : allResults) {
          for (var result : fileResults) {
            System.out.printf(Locale.ROOT, "%-23s   %-16f   %s\n",
                result.label(),
                result.score(),
                Arrays.toString(result.coordinate()));
          }
        }
      }
    }
  }

}

```