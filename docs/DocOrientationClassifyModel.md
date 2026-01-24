
## DocOrientationClassifyModel

### Performance
| Model                 |  Type   |  CUDA   |
|:----------------------|:-------:|:-------:|
| PP-LCNet_x1_0_doc_ori | End2End | 61.97ms |
| PP-LCNet_x1_0_doc_ori |  Infer  | 5.48ms  |


### Test Environment
- GPU: NVIDIA RTX 4060 Ti
- CPU: Intel i5-14600KF
- OS: Windows 11 x64
- Batch size: 16 (for batchPredict)
- DPI: 144
- PDF: 16 pages
- Iterations: 1234
- Date: 2026-01-24

### Performance Metrics
- **End2End**: Includes PDF loading, page-to-image conversion, image preprocessing, model inference, and postprocessing.
- **Infer**: Model inference (including I/O and postprocessing).

**Results are averaged over 16 pages and 1234 iterations.**


### Code Example

```java
static void main() throws Exception {
    String pdfFile = "D:\\papers\\1706.03762_zh_CN.pdf";
    String rootDir = "D:\\models\\docOrient";
    int dpi = 144;
    int num = 1;
    try (
            var env = OrtEnvironment.getEnvironment();
            var docOriModel = new DocOrientationClassifyModel(
                    rootDir,
                    "PP-LCNet_x1_0_doc_ori",
                    env,
                    0
            )
    ) {
        long inferTotal = 0;
        int imageCount = 1;
        LocalDateTime start = LocalDateTime.now();
        for (int x = 0; x < num; x++) {
            try (
                 var document = Loader.loadPDF(new File(pdfFile));
                 var matManager = new MatManager();
                 var ndManager = NDManager.newBaseManager()
            ) {
                int endPage = document.getNumberOfPages();
                imageCount = endPage;
                PDFRenderer renderer = new PDFRenderer(document);

                List<PreProcessResult> pprs = new ArrayList<>();
                for (int i = 0; i < endPage; i++) {
                    BufferedImage bufferedImage = renderer.renderImageWithDPI(i, dpi, ImageType.RGB);
                    Mat mat = OpenCVImage.image2Mat(matManager, bufferedImage);
                    Mat rgbMat = ImageUtil.bgrToRgb(matManager, mat);
                    pprs.add(docOriModel.processRgb(matManager, rgbMat, ndManager));
                }

                LocalDateTime inferStart = LocalDateTime.now();
                List<ClassificationResult> results = docOriModel.batchPredict(pprs, 16, matManager, ndManager, Map.of());
                LocalDateTime inferEnd = LocalDateTime.now();
                inferTotal += Duration.between(inferStart, inferEnd).toNanos();
                if (x == 0) {
                    System.out.println("\n".repeat(3));
                    for (int i = 0; i < endPage; i++) {
                        System.out.println("Page: " + String.format("%3d", i + 1)
                                + " ".repeat(5) + String.format("%6s", results.get(i).label())
                                + " ".repeat(5) + results.get(i).score());
                    }
                } else {
                    System.out.println("iter: " + String.format("%6d", x) + " ".repeat(7) + LocalDateTime.now());
                }
            }
        }
        LocalDateTime end = LocalDateTime.now();
        System.out.println("End2End: " + (Duration.between(start, end).toNanos() / 1000_000d / (double) num / (double) imageCount) + "ms");
        System.out.println("infer: " + (inferTotal / 1000_000d / (double) num / (double) imageCount) + "ms");
        System.out.println(start + "\t\t" + end);
    }
}

```
