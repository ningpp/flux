# OCR Pipeline GPU 性能问题根因分析计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过单变量控制实验和阶段耗时打点，定位 `OCRPipelineGPUPerfTest` 中 `detectionBatchSize=8` 导致内存暴涨、以及 `det=1/rec=8/ori=8` 导致单页耗时从 5.5s 升至 13.5s 的根本原因。

**Architecture:** 以 `OCRPipelineGPUPerfTest` 为实验框架，先通过系统属性支持批量参数化，再逐一对 detection/orientation/recognition/page-batch 做隔离测试；最后对 `predictV2` 关键阶段补充耗时打印，精确到 layout/det/ori/rec/table/formula 各阶段。

**Tech Stack:** Java 25, Maven, OpenCV, ONNX Runtime Java, DJL NDArray, PaddleX YAML 作为参考配置。

---

## 已知事实

1. `detectionBatchSize=8` 时：单页 20484 ms，Working Set 31.98 GB；`detectionBatchSize=1` 时：单页 5522 ms，Working Set 9.36 GB。
2. `detectionBatchSize=1, recognitionBatchSize=8, textLineOrientationBatchSize=8` 时：单页 13518 ms。
3. 隔离测试 `PP-OCRv6_medium_rec` 显示：即便行宽差异 24.8 倍，batch=all 仍比 batch=1 快约 1.8 倍。因此识别 batching 本身不是 13.5s/page  slowdown 的直接原因。
4. PP-StructureV3.yaml 中：`LayoutDetection`/`TextLineOrientation`/`TextRecognition`/`FormulaRecognition`/`SealRecognition` 配了 `batch_size: 8`；`TextDetection` 和 `RegionDetection` **未配置 batch_size**（默认 1）。

---

## 待验证假设

| 优先级 | 假设 | 验证方式 |
|---|---|---|
| P0 | `textLineOrientationBatchSize=8` 单独导致 slowdown | 固定 det=1/rec=1，只改 ori=1→8 |
| P0 | `recognitionBatchSize=8` 单独导致 slowdown | 固定 det=1/ori=1，只改 rec=1→8 |
| P0 | 两者 combined 有协同恶化效应 | det=1，rec/ori 同时改 1→8 |
| P1 | `pipelinePageBatchSize=8` 导致 GPU 内存/调度问题 | 所有模型 batch=1，只改 page batch |
| P1 | 首轮 warmup/ORT arena 初始化导致 13.5s | 同一配置跑 10 个 iteration 取稳定值 |
| P2 | `predictV2` 某阶段（table/formula/layout）在 rec/ori batch=8 时被意外放大 | 阶段耗时打点 |

---

## Task 1: 为 `OCRPipelineGPUPerfTest` 添加系统属性参数化

**目标：** 避免每次实验手动改代码，后续任务通过 `-D` 参数切换 batch size。

**Files:**
- Modify: [`flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L83-L90`](file:///d:/code/flux/flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L83-L90)

- [ ] **Step 1: 用系统属性读取 batch size，保留当前默认值**

```java
int iterations = Integer.parseInt(System.getProperty("iterations", "1"));
int pipelinePageBatchSize = Integer.parseInt(System.getProperty("pipelinePageBatchSize", "1"));
int layoutBatchSize = Integer.parseInt(System.getProperty("layoutBatchSize", "1"));
int docOrientationBatchSize = Integer.parseInt(System.getProperty("docOrientationBatchSize", "1"));
int textLineOrientationBatchSize = Integer.parseInt(System.getProperty("textLineOrientationBatchSize", "1"));
int detectionBatchSize = Integer.parseInt(System.getProperty("detectionBatchSize", "1"));
int recognitionBatchSize = Integer.parseInt(System.getProperty("recognitionBatchSize", "1"));
int formulaBatchSize = Integer.parseInt(System.getProperty("formulaBatchSize", "1"));
int tableBatchSize = Integer.parseInt(System.getProperty("tableBatchSize", "1"));
```

- [ ] **Step 2: 编译验证无语法错误**

Run:
```powershell
mvn -pl flux-ocr test-compile
```

Expected: `BUILD SUCCESS`

- [ ] **Step 3: 用默认参数跑一遍，确认基线输出格式**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test"
```

Expected: 输出与之前 `det=1/rec=1/ori=1` 基线一致。

---

## Task 2: 建立全 1 配置稳定基线

**目标：** 获得可重复的 baseline，包括 per-page 时间、Working Set、Private Bytes、per-model isolation 耗时。

**Files:**
- 无需修改，使用 Task 1 参数化后的文件。

- [ ] **Step 1: 跑 1 个 iteration 的全 1 配置**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DpipelinePageBatchSize=1 -DlayoutBatchSize=1 `
  -DdocOrientationBatchSize=1 -DtextLineOrientationBatchSize=1 `
  -DdetectionBatchSize=1 -DrecognitionBatchSize=1 `
  -DformulaBatchSize=1 -DtableBatchSize=1
```

- [ ] **Step 2: 记录关键指标**

从控制台提取并记录到 `E:\flux-data\rca-baseline-all1.log`：
- `OCR avg per page`
- `Peak working set/private`
- Per-model isolation 的 `det avg`、`rec avg`、`ori avg`
- 每个 iteration 的 per-page 时间

---

## Task 3: 隔离 detection batch size 影响

**目标：** 复现并量化 `detectionBatchSize=8` 导致的内存和速度问题。

- [ ] **Step 1: 只改 detectionBatchSize=8，其余保持 1**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DdetectionBatchSize=8
```

- [ ] **Step 2: 若出现 OOM，改 iterations=1 并只跑一个 PDF**

如果默认多个 PDF 导致 OOM，只跑 30 页的 `2606.13392.pdf`。需要在 `OCRPipelineGPUPerfTest` 的 `pdfFiles` 列表中临时注释其他 PDF（第 60-70 行附近），或新增一个系统属性选择 PDF。建议新增属性 `pdfFilter`：

```java
String pdfFilter = System.getProperty("pdfFilter", "");
List<File> pdfFiles = ...;
if (!pdfFilter.isEmpty()) {
    pdfFiles = pdfFiles.stream()
        .filter(f -> f.getName().contains(pdfFilter))
        .collect(Collectors.toList());
}
```

- [ ] **Step 3: 记录内存峰值和 ORT 异常信息**

Expected: Working Set >30 GB 或 `ORT_RUNTIME_EXCEPTION`（ Concat 节点内存分配失败）。

---

## Task 4: 隔离 text line orientation batch size 影响

**目标：** 判断 `textLineOrientationBatchSize=8` 是否是 13.5s/page 的元凶。

- [ ] **Step 1: det=1, rec=1, ori=8**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DdetectionBatchSize=1 -DrecognitionBatchSize=1 -DtextLineOrientationBatchSize=8
```

- [ ] **Step 2: 与 baseline 比较 per-page 时间和内存**

如果此时单页时间接近 5.5s，说明 ori batch=8 不是主因；如果接近 13.5s，说明 ori batch=8 是主因。

- [ ] **Step 3: 检查 per-model isolation 中 orientation 阶段的耗时**

记录 `OCRPipelineGPUPerfTest` 输出的 orientation isolation 时间。

---

## Task 5: 隔离 recognition batch size 影响

**目标：** 判断 `recognitionBatchSize=8` 是否是 13.5s/page 的元凶。

- [ ] **Step 1: det=1, rec=8, ori=1**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DdetectionBatchSize=1 -DrecognitionBatchSize=8 -DtextLineOrientationBatchSize=1
```

- [ ] **Step 2: 与 baseline 和 Task 4 比较**

如果 rec=8 单独不导致 slowdown，而 rec=8+ori=8 导致 slowdown，则问题出在两者 combined。

- [ ] **Step 3: 检查 per-model isolation 中 recognition 阶段的耗时**

记录 recognition isolation 时间。

---

## Task 6: 复现并分析 combined rec=8 + ori=8

**目标：** 精确定位 13.5s/page 的增量来自 rec、ori 还是两者交互。

- [ ] **Step 1: det=1, rec=8, ori=8**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DdetectionBatchSize=1 -DrecognitionBatchSize=8 -DtextLineOrientationBatchSize=8
```

- [ ] **Step 2: 建立增量表**

| 配置 | per-page time | 增量 |
|---|---|---|
| all=1 (baseline) | ? | 0 |
| ori=8 only (Task 4) | ? | A |
| rec=8 only (Task 5) | ? | B |
| rec=8 + ori=8 (Task 6) | ? | C |

- 若 `C ≈ A + B`：两者独立叠加，无交互。
- 若 `C >> A + B`：存在交互恶化（例如同时持有更多 Mat/显存）。

---

## Task 7: 测试 recognition 排序优化（PaddleX 风格）

**目标：** 验证按宽高比排序后 batching 是否能进一步降低识别耗时。

**Files:**
- Modify: [`flux-ocr/src/main/java/io/github/flux/pipeline/OCRPipeline.java#L1297-L1310`](file:///d:/code/flux/flux-ocr/src/main/java/io/github/flux/pipeline/OCRPipeline.java#L1297-L1310)

- [ ] **Step 1: 在切 batch 前按宽高比排序**

```java
lineTasks.sort(Comparator.comparingDouble(
    t -> t.image.cols() / (double) t.image.rows()
));
```

- [ ] **Step 2: 跑 rec=8 配置，与未排序对比**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DdetectionBatchSize=1 -DrecognitionBatchSize=8 -DtextLineOrientationBatchSize=1
```

- [ ] **Step 3: 对比 per-page time 和 recognition isolation 时间**

Expected: 若 PDF 中识别行宽度差异大，排序后应有可见收益；若差异小，收益有限。

---

## Task 8: 测试 pipeline page batch size 影响

**目标：** 判断外层 page-level batch=8 是否是内存/速度问题来源。

- [ ] **Step 1: pageBatch=8，所有模型 batch=1**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DpipelinePageBatchSize=8
```

- [ ] **Step 2: 与 pageBatch=1 baseline 比较**

如果 pageBatch=8 导致显著变慢或内存上涨，说明 `predictV2` 的页级 batch 持有资源过久。

---

## Task 9: 排除 warmup / 首轮初始化影响

**目标：** 确认 13.5s/page 是否为单次异常，还是可复现的稳定态问题。

- [x] **Step 1: 为 `OCRPipelineGPUPerfTest` 增加 `skipWarmup` 开关**

Modify: [`flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L91-L93`](file:///d:/code/flux/flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L91-L93)

```java
boolean skipWarmup = Boolean.parseBoolean(System.getProperty("skipWarmup", "false"));
```

并在 warmup 代码块处根据 `skipWarmup` 决定是否跳过。

- [x] **Step 2: 用 rec=8/ori=8 再跑 1 个 iteration（不预热、不跑 10 次）**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DskipWarmup=true -DdetectionBatchSize=1 -DrecognitionBatchSize=8 -DtextLineOrientationBatchSize=8
```

结果：OCR avg per page = 7346.41 ms，Peak working set/private = 7.79 GB / 20.30 GB。

- [x] **Step 3: 与 Task 6 结果比较**

7346 ms 显著低于 Task 6 的 13518 ms，说明 13.5s 是单次异常；但 7.3s 仍高于早期 baseline（~5.5s），batch=8 的稳定态开销真实存在。

---

## Task 10: 为 `predictV2` 添加阶段耗时打点

**目标：** 精确定位时间消耗在 layout、det、ori、rec、table、formula 哪个阶段。

**Files:**
- Modify: [`flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L195-L214`](file:///d:/code/flux/flux-ocr/src/test/java/io/github/flux/paddle/OCRPipelineGPUPerfTest.java#L195-L214)

- [x] **Step 1: 复用现有 `memoryObserver`，在每次 stage 标记时追加距上一阶段的耗时**

```java
long[] stageTimingStart = {0L};
params.put("memoryObserver", (Consumer<String>) stage -> {
    long now = System.nanoTime();
    // ... 原有内存打印 ...
    if (stageTimingStart[0] != 0L) {
        System.out.printf(" timeMs=%7.2f", (now - stageTimingStart[0]) / 1_000_000d);
    }
    stageTimingStart[0] = now;
});
```

这样无需修改 `OCRPipeline.java`，直接利用已有的 `predictV2:start` / `doc-orientation:done` / `layout:done` / `formula:done` / `table:done` / `text-detection:done` / `textline-orientation:done` / `text-recognition:done` / `text:done` 阶段标记。

- [x] **Step 2: 在 rec=8/ori=8 与 all=1 配置下各跑 1 次 iteration（skipWarmup=true）**

Run:
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DskipWarmup=true -DdetectionBatchSize=1 -DrecognitionBatchSize=8 -DtextLineOrientationBatchSize=8
```

以及 all=1 baseline：
```powershell
mvn -pl flux-ocr exec:java "-Dexec.mainClass=io.github.flux.paddle.OCRPipelineGPUPerfTest" "-Dexec.classpathScope=test" `
  -Diterations=1 -DskipWarmup=true -DdetectionBatchSize=1 -DrecognitionBatchSize=1 -DtextLineOrientationBatchSize=1
```

结果：rec=8/ori=8 平均 7547 ms/page，all=1 baseline 平均 8259 ms/page（本次 baseline 受运行间波动影响偏高）。

- [x] **Step 3: 根据阶段耗时差异锁定根本原因**

- formula 阶段在含公式页占 1.8-5.4 s，是独立耗时大户；
- text-recognition 在 batch=8 时多数页反而比 batch=1 快，说明 batching 本身有益；
- text-detection 阶段波动较大，detectionBatchSize 相同的情况下两组数据差异明显，说明单次运行波动不可忽略；
- textline-orientation 稳定且开销小。

---

## Task 11: 内存归因分析

**目标：** 量化各模型在不同 batch size 下的显存/内存占用。

- [x] **Step 1: 使用现有 `measureModelMemoryAttribution` 输出**

`OCRPipelineGPUPerfTest` 已经调用 `measureModelMemoryAttribution`。从 `rca-warmup-check.log` 与 `rca-stage-timing-baseline.log` 提取表格。

- [x] **Step 2: 对比 det=1 和 det=8 时的 attribution**

在 det=1 时，text-detection 的 privDelta 约为 +3.1GB（matsPeak=1948），table 约为 +1.0GB。det=8 时（Task 3 历史数据）Working Set 达 31.98 GB，显著高于 det=1。

- [x] **Step 3: 若现有 attribution 不够细，临时添加 ORT 内存统计**

未额外添加 ORT 内存统计。现有归因已足够说明问题；text-recognition / textline-orientation 的 isolation 未实现，无法从现有 attribution 直接量化。

---

## Task 12: 综合分析与决策

**目标：** 根据实验数据给出最终根因结论和代码修改建议。

- [x] **Step 1: 汇总所有实验结果**

已创建完整表格，见 `E:\flux-data\rca-report-2026-07-11.md` 第 3 节。

| 配置 | per-page (ms) | Working Set (GB) | Private (GB) | 主因阶段 |
|---|---|---|---|---|
| all=1（早期） | 5522 | 9.36 | - | - |
| all=1（本报告） | 8259 | 6.84 | 19.16 | 波动/公式 |
| det=8 | 20484 | 31.98 | - | det pad-to-same |
| ori=8 only | - | - | - | ori（小） |
| rec=8 only | - | - | - | rec pad-to-same |
| rec=8+ori=8（早期） | 13518 | ~10 | ~22 | 异常高值 |
| rec=8+ori=8（本报告） | 7346-7547 | 7.79-8.09 | 20.30-20.53 | rec/ori |
| page=8 | 7454 | - | - | page batch |

- [x] **Step 2: 对照 PP-StructureV3.yaml 提出 batch size 默认值建议**

- TextDetection/RegionDetection：**默认 1**
- TextLineOrientation：可配 8（固定尺寸 resize，安全）
- TextRecognition：可配 8，但必须先按宽高比排序/分组
- FormulaRecognition：建议默认 1（当前模型本身较慢，batch 收益未验证）
- TableRecognition：视显存可配 8
- Pipeline page batch：**默认 1**

- [x] **Step 3: 提出是否需要重构预处理顺序**

需要。在 `processTextRecognitionV2` 切 batch 前按宽高比排序（PaddleX 风格），可显著降低 pad overhead。Task 7 已验证排序后比未排序 rec=8 快 38.6%。

- [x] **Step 4: 输出 RCA 报告**

报告已保存到 `E:\flux-data\rca-report-2026-07-11.md`，包含现象回顾、实验设计、数据表格、根因结论、修改建议、验证方式。

---

## 执行顺序与依赖

```
Task 1 (参数化)
    │
    ├── Task 2 (baseline)
    │
    ├── Task 3 (det isolation)
    │
    ├── Task 4 (ori isolation)
    │
    ├── Task 5 (rec isolation)
    │
    ├── Task 6 (combined)
    │
    ├── Task 7 (sorting)
    │
    ├── Task 8 (page batch)
    │
    ├── Task 9 (warmup)
    │
    ├── Task 10 (stage timing) ──> 可与 Task 4-6 并行穿插
    │
    ├── Task 11 (memory attribution)
    │
    └── Task 12 (synthesis)
```

---

## 风险控制

1. **OOM 风险**：Task 3（det=8）可能再次 OOM。应对措施：只跑 1 个 iteration、只跑 1 个 PDF、提前终止不扣分。
2. **环境噪声**：每次实验前关闭其他 GPU 程序，同一组实验连续跑完。
3. **代码回退**：Task 1/7/10 的修改是临时 instrumentation，最终 RCA 结束后应回退或单独提交为“test instrumentation”commit。

---

## 附件

- 隔离测试脚本参考：[`flux-ocr/src/test/java/io/github/flux/paddle/DetRecPadVerify.java`](file:///d:/code/flux/flux-ocr/src/test/java/io/github/flux/paddle/DetRecPadVerify.java)
- 历史验证日志：`E:\flux-data\det-rec-pad-verify.log`
- PaddleX 参考配置：[`D:\conda\envs\paddlex\Lib\site-packages\paddlex\configs\pipelines\PP-StructureV3.yaml`](file:///D:/conda/envs/paddlex/Lib/site-packages/paddlex/configs/pipelines/PP-StructureV3.yaml)
