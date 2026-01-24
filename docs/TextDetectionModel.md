

## TextDetectionModel

### Performance
| Model               |  Type   |  CUDA   |
|:--------------------|:-------:|:-------:|
| PP-OCRv5_server_det | End2End | 92.41ms |
| PP-OCRv5_server_det |  Infer  | 81.74ms |
| PP-OCRv5_mobile_det | End2End | 69.20ms |
| PP-OCRv5_mobile_det |  Infer  | 58.36ms |

### Test Environment
- GPU: NVIDIA RTX 4060 Ti
- CPU: Intel i5-14600KF
- OS: Windows 11 x64
- Batch size: 3 (for batchPredict)
- Image: 3 images
- Iterations: 1234
- Date: 2026-01-24

### Performance Metrics
- **End2End**: Includes image read, image preprocessing, model inference, and postprocessing.
- **Infer**: Model inference (including I/O and postprocessing).

**Results are averaged over 3 pages and 1234 iterations.**


### Code Example

```java
static void main() throws Exception {
    String[] imgFiles = {
            "D:\\tmp\\img-2026-01-24-193955.png",
            "D:\\papers\\imgs\\1706.03762_zh_CN\\00001.png",
            "D:\\papers\\imgs\\1706.03762_zh_CN-DPI144\\00001.png"
    };
    String rootDir = "D:\\models\\text-det";
    int num = 3456;
    int imageCount = imgFiles.length;
    try (
            var env = OrtEnvironment.getEnvironment();
            var detModel = new TextDetectionModel(
                    rootDir,
                    "PP-OCRv5_mobile_det",
                    env,
                    0
            )
    ) {
        long inferTotal = 0;
        LocalDateTime start = LocalDateTime.now();
        List<TextDetectionResult> results = null;
        for (int x = 0; x < num; x++) {
            try (
                    var matManager = new MatManager();
                    var ndManager = NDManager.newBaseManager()
            ) {
                List<PreProcessResult> pprs = new ArrayList<>();
                for (String img : imgFiles) {
                    pprs.add(detModel.process(matManager, img, ndManager));
                }

                LocalDateTime inferStart = LocalDateTime.now();
                results = detModel.batchPredict(pprs, imageCount, matManager, ndManager, Map.of());
                LocalDateTime inferEnd = LocalDateTime.now();
                inferTotal += Duration.between(inferStart, inferEnd).toNanos();
                if (x == 0) {
                    System.out.println("\n".repeat(3));
                    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(results));
                } else {
                    System.out.println("iter: " + String.format("%6d", x) + " ".repeat(7) + LocalDateTime.now());
                }
            }
        }
        LocalDateTime end = LocalDateTime.now();
        System.out.println("End2End: " + (Duration.between(start, end).toNanos() / 1000_000d / (double) num / (double) imageCount) + "ms");
        System.out.println("infer: " + (inferTotal / 1000_000d / (double) num / (double) imageCount) + "ms");
        System.out.println(start + "\t\t" + end);
        System.out.println("\n".repeat(11));


        IFontResource fontResource = EFontResourceOpenSans.OPEN_SANS_NORMAL.getFontResource();
        Font font = Font.createFonts(fontResource.getInputStream())[0].deriveFont(Font.PLAIN, 18f);
        for (int i = 0; i < imgFiles.length; i++) {
            String img = imgFiles[i];
            BufferedImage bufferedImage = ImageIO.read(new File(img));
            Graphics2D g2d = bufferedImage.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setFont(font);
            g2d.setColor(Color.BLUE);

            TextDetectionResult tdr = results.get(i);
            int index = 0;
            for (int[][] poly : tdr.polys()) {
                for (int j = 0; j < poly.length; j++) {
                    if (j == poly.length - 1) {
                        g2d.drawLine(poly[j][0], poly[j][1], poly[0][0], poly[0][1]);
                    } else {
                        g2d.drawLine(poly[j][0], poly[j][1], poly[j+1][0], poly[j+1][1]);
                    }
                }
                g2d.drawString(String.valueOf(tdr.scores().get(index)), poly[0][0], poly[0][1]);
                index++;
            }

            g2d.dispose();
            ImageIO.write(bufferedImage, "png", new File("D:\\tmp\\text-det-" + new File(img).getName() + "-" + UUID.randomUUID() + ".png"));
        }
    }
}
```
