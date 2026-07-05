package io.github.flux.falconocr;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import io.github.flux.core.MatManager;
import io.github.flux.exception.FluxException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class FalconOcrProcessor {

    static final int PAD_TOKEN_ID = 0;
    static final int EOS_TOKEN_ID = 11;
    static final int END_OF_QUERY_TOKEN_ID = 263;
    static final int IMG_ID = 227;
    static final int IMAGE_CLS_TOKEN_ID = 244;
    static final int IMAGE_REG_1_TOKEN_ID = 245;
    static final int IMAGE_REG_2_TOKEN_ID = 246;
    static final int IMAGE_REG_3_TOKEN_ID = 247;
    static final int IMAGE_REG_4_TOKEN_ID = 248;
    static final int IMG_END_ID = 230;

    static final int PATCH_SIZE = 16;
    static final int IMAGE_PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3;
    static final int MIN_DIMENSION = 64;
    static final int MAX_DIMENSION = 1024;

    private FalconOcrProcessor() {
    }

    record Preprocessed(
            long[] tokens,
            float[] pixels,
            int height,
            int width,
            int patchRows,
            int patchCols,
            String category
    ) {
        int promptLength() {
            return tokens.length;
        }

        int imageTokenCount() {
            return patchRows * patchCols;
        }
    }

    record BatchPreprocessed(
            long[][] batchTokens,
            float[][][] batchImagePatches,
            long[][] batchPosT,
            float[][][] batchPosHw,
            boolean[][][] batchAttentionMask,
            int[] promptLengths,
            int paddedPromptLength
    ) {
        int batchSize() {
            return batchTokens.length;
        }
    }

    static Preprocessed process(MatManager matManager,
                                Mat rgbMat,
                                HuggingFaceTokenizer tokenizer,
                                String category) {
        String checkedCategory = requireCategory(category);
        Mat normalized = null;
        Mat bounded = resizeImageIfNecessary(matManager, rgbMat);
        Mat smart = null;
        try {
            smart = smartResize(matManager, bounded);
            int height = smart.rows();
            int width = smart.cols();
            int patchRows = height / PATCH_SIZE;
            int patchCols = width / PATCH_SIZE;
            long[] tokens = buildInputIds(tokenizer, checkedCategory, patchRows * patchCols);

            normalized = matManager.newMat();
            smart.convertTo(normalized, CvType.CV_32FC3, 1.0 / 127.5, -1.0);
            float[] pixels = new float[height * width * 3];
            normalized.get(0, 0, pixels);

            return new Preprocessed(tokens, pixels, height, width, patchRows, patchCols, checkedCategory);
        } finally {
            if (bounded != rgbMat) {
                matManager.release(bounded);
            }
            if (smart != null && smart != bounded) {
                matManager.release(smart);
            }
            if (normalized != null) {
                matManager.release(normalized);
            }
        }
    }

    static BatchPreprocessed batchPad(List<Preprocessed> items) {
        if (items == null || items.isEmpty()) {
            throw new FluxException("Falcon-OCR batch must not be empty");
        }

        int batch = items.size();
        int maxSeq = items.stream().mapToInt(i -> i.tokens.length).max().orElseThrow();
        int maxHeight = items.stream().mapToInt(Preprocessed::height).max().orElseThrow();
        int maxWidth = items.stream().mapToInt(Preprocessed::width).max().orElseThrow();

        long[][] tokens = new long[batch][maxSeq];
        long[][] posT = new long[batch][maxSeq];
        float[][][] posHw = new float[batch][maxSeq][2];
        float[][][] imagePatches = new float[batch][maxSeq][IMAGE_PATCH_DIM];
        int[] promptLengths = new int[batch];

        for (int b = 0; b < batch; b++) {
            Preprocessed item = items.get(b);
            promptLengths[b] = item.tokens.length;
            int offset = maxSeq - item.tokens.length;
            Arrays.fill(tokens[b], PAD_TOKEN_ID);
            fillNaN(posHw[b]);
            System.arraycopy(item.tokens, 0, tokens[b], offset, item.tokens.length);
        }

        float[][][][] paddedPixels = buildPaddedPixels(items, maxHeight, maxWidth);
        boolean[][][] pixelMask = buildPixelMask(items, maxHeight, maxWidth);
        buildImagePatches(tokens, imagePatches, paddedPixels, pixelMask, maxHeight, maxWidth);
        buildPositions(tokens, posT, posHw, pixelMask, maxHeight, maxWidth);
        boolean[][][] attentionMask = buildAttentionMask(tokens);

        return new BatchPreprocessed(
                tokens,
                imagePatches,
                posT,
                posHw,
                attentionMask,
                promptLengths,
                maxSeq
        );
    }

    private static String requireCategory(String category) {
        if ("formula".equals(category) || "table".equals(category)) {
            return category;
        }
        throw new FluxException("Falcon-OCR only supports formula and table, got: " + category);
    }

    private static Mat resizeImageIfNecessary(MatManager matManager, Mat rgbMat) {
        int originalWidth = rgbMat.cols();
        int originalHeight = rgbMat.rows();
        double aspectRatio = (double) originalWidth / originalHeight;

        if (MIN_DIMENSION <= originalWidth && originalWidth <= MAX_DIMENSION
                && MIN_DIMENSION <= originalHeight && originalHeight <= MAX_DIMENSION) {
            return rgbMat;
        }

        int newWidth;
        int newHeight;
        boolean vertical = originalWidth < originalHeight;
        if (originalWidth < MIN_DIMENSION || originalHeight < MIN_DIMENSION) {
            if (vertical) {
                newWidth = MIN_DIMENSION;
                newHeight = (int) (newWidth / aspectRatio);
            } else {
                newHeight = MIN_DIMENSION;
                newWidth = (int) (newHeight * aspectRatio);
            }
        } else {
            if (vertical) {
                newWidth = MAX_DIMENSION;
                newHeight = (int) (newWidth / aspectRatio);
            } else {
                newHeight = MAX_DIMENSION;
                newWidth = (int) (newHeight * aspectRatio);
            }
        }

        if (newWidth > MAX_DIMENSION) {
            newWidth = MAX_DIMENSION;
            newHeight = (int) (newWidth / aspectRatio);
        }
        if (newHeight > MAX_DIMENSION) {
            newHeight = MAX_DIMENSION;
            newWidth = (int) (newHeight * aspectRatio);
        }

        return resizeRgb(matManager, rgbMat, newWidth, newHeight);
    }

    private static Mat smartResize(MatManager matManager, Mat image) {
        int height = image.rows();
        int width = image.cols();
        int hBar = Math.round((float) height / PATCH_SIZE) * PATCH_SIZE;
        int wBar = Math.round((float) width / PATCH_SIZE) * PATCH_SIZE;
        int minPixels = 56 * 56;
        int maxPixels = 28 * 28 * 1280;

        if ((long) hBar * wBar > maxPixels) {
            double beta = Math.sqrt((double) height * width / maxPixels);
            hBar = (int) Math.floor(height / beta / PATCH_SIZE) * PATCH_SIZE;
            wBar = (int) Math.floor(width / beta / PATCH_SIZE) * PATCH_SIZE;
        } else if ((long) hBar * wBar < minPixels) {
            double beta = Math.sqrt((double) minPixels / (height * width));
            hBar = (int) Math.ceil(height * beta / PATCH_SIZE) * PATCH_SIZE;
            wBar = (int) Math.ceil(width * beta / PATCH_SIZE) * PATCH_SIZE;
        }

        if (hBar == height && wBar == width) {
            return image;
        }

        return resizeRgb(matManager, image, wBar, hBar);
    }

    private static Mat resizeRgb(MatManager matManager, Mat rgbMat, int width, int height) {
        return resizeRgbPillowStyle(matManager, rgbMat, width, height);
    }

    private static Mat resizeRgbPillowStyle(MatManager matManager, Mat rgbMat, int width, int height) {
        int srcWidth = rgbMat.cols();
        int srcHeight = rgbMat.rows();
        byte[] src = new byte[srcWidth * srcHeight * 3];
        rgbMat.get(0, 0, src);

        float[] tmp = new float[height * srcWidth * 3];
        double scaleY = (double) srcHeight / height;
        for (int y = 0; y < height; y++) {
            double center = (y + 0.5) * scaleY;
            double filterScale = Math.max(scaleY, 1.0);
            double support = 2.0 * filterScale;
            int xmin = Math.max(0, (int) Math.floor(center - support + 0.5));
            int xmax = Math.min(srcHeight, (int) Math.floor(center + support + 0.5));
            double[] weights = normalizedWeights(center, xmin, xmax, filterScale);
            for (int x = 0; x < srcWidth; x++) {
                for (int c = 0; c < 3; c++) {
                    double value = 0.0;
                    for (int yy = xmin; yy < xmax; yy++) {
                        int srcIdx = (yy * srcWidth + x) * 3 + c;
                        value += (src[srcIdx] & 0xFF) * weights[yy - xmin];
                    }
                    tmp[(y * srcWidth + x) * 3 + c] = (float) value;
                }
            }
        }

        byte[] out = new byte[width * height * 3];
        double scaleX = (double) srcWidth / width;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double center = (x + 0.5) * scaleX;
                double filterScale = Math.max(scaleX, 1.0);
                double support = 2.0 * filterScale;
                int xmin = Math.max(0, (int) Math.floor(center - support + 0.5));
                int xmax = Math.min(srcWidth, (int) Math.floor(center + support + 0.5));
                double[] weights = normalizedWeights(center, xmin, xmax, filterScale);
                for (int c = 0; c < 3; c++) {
                    double value = 0.0;
                    for (int xx = xmin; xx < xmax; xx++) {
                        value += tmp[(y * srcWidth + xx) * 3 + c] * weights[xx - xmin];
                    }
                    out[(y * width + x) * 3 + c] = (byte) clampToByte(value);
                }
            }
        }

        Mat mat = matManager.newMat(height, width, CvType.CV_8UC3);
        mat.put(0, 0, out);
        return mat;
    }

    private static double[] normalizedWeights(double center,
                                              int startInclusive,
                                              int endExclusive,
                                              double filterScale) {
        double[] weights = new double[endExclusive - startInclusive];
        double sum = 0.0;
        for (int i = startInclusive; i < endExclusive; i++) {
            double weight = bicubicFilter((i + 0.5 - center) / filterScale);
            weights[i - startInclusive] = weight;
            sum += weight;
        }
        if (sum != 0.0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] /= sum;
            }
        }
        return weights;
    }

    private static double bicubicFilter(double x) {
        double abs = Math.abs(x);
        if (abs < 1.0) {
            return ((1.5 * abs - 2.5) * abs) * abs + 1.0;
        }
        if (abs < 2.0) {
            return (((-0.5 * abs + 2.5) * abs - 4.0) * abs) + 2.0;
        }
        return 0.0;
    }

    private static int clampToByte(double value) {
        int rounded = (int) Math.round(value);
        if (rounded < 0) {
            return 0;
        }
        if (rounded > 255) {
            return 255;
        }
        return rounded;
    }

    private static long[] buildInputIds(HuggingFaceTokenizer tokenizer, String category, int imageTokenCount) {
        String prompt = "<|image|>" + promptFor(category) + "\n<|OCR_PLAIN|>";
        String imageToken = "<|image|>";
        int split = prompt.indexOf(imageToken);
        long[] prefix = tokenizer.encode(prompt.substring(0, split)).getIds();
        long[] suffix = tokenizer.encode(prompt.substring(split + imageToken.length())).getIds();

        List<Long> ids = new ArrayList<>();
        for (long id : prefix) {
            ids.add(id);
        }
        ids.add((long) IMAGE_CLS_TOKEN_ID);
        ids.add((long) IMAGE_REG_1_TOKEN_ID);
        ids.add((long) IMAGE_REG_2_TOKEN_ID);
        ids.add((long) IMAGE_REG_3_TOKEN_ID);
        ids.add((long) IMAGE_REG_4_TOKEN_ID);
        for (int i = 0; i < imageTokenCount; i++) {
            ids.add((long) IMG_ID);
        }
        ids.add((long) IMG_END_ID);
        for (long id : suffix) {
            ids.add(id);
        }
        long[] out = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    private static String promptFor(String category) {
        if ("formula".equals(category)) {
            return "Extract the formula content from this image.";
        }
        if ("table".equals(category)) {
            return "Extract the table content from this image.";
        }
        throw new FluxException("Falcon-OCR only supports formula and table, got: " + category);
    }

    private static float[][][][] buildPaddedPixels(List<Preprocessed> items, int maxHeight, int maxWidth) {
        float[][][][] out = new float[items.size()][maxHeight][maxWidth][3];
        for (int b = 0; b < items.size(); b++) {
            Preprocessed item = items.get(b);
            for (int y = 0; y < item.height; y++) {
                for (int x = 0; x < item.width; x++) {
                    int src = (y * item.width + x) * 3;
                    out[b][y][x][0] = item.pixels[src];
                    out[b][y][x][1] = item.pixels[src + 1];
                    out[b][y][x][2] = item.pixels[src + 2];
                }
            }
        }
        return out;
    }

    private static boolean[][][] buildPixelMask(List<Preprocessed> items, int maxHeight, int maxWidth) {
        boolean[][][] out = new boolean[items.size()][maxHeight][maxWidth];
        for (int b = 0; b < items.size(); b++) {
            Preprocessed item = items.get(b);
            for (int y = 0; y < item.height; y++) {
                Arrays.fill(out[b][y], 0, item.width, true);
            }
        }
        return out;
    }

    private static void buildImagePatches(long[][] tokens,
                                          float[][][] imagePatches,
                                          float[][][][] paddedPixels,
                                          boolean[][][] pixelMask,
                                          int maxHeight,
                                          int maxWidth) {
        int expectedPatches = countTokens(tokens, IMG_ID);
        int written = 0;
        int patchRows = maxHeight / PATCH_SIZE;
        int patchCols = maxWidth / PATCH_SIZE;
        for (int b = 0; b < tokens.length; b++) {
            int tokenPos = firstImageTokenIndex(tokens[b]);
            for (int pr = 0; pr < patchRows; pr++) {
                for (int pc = 0; pc < patchCols; pc++) {
                    if (!patchHasValidPixel(pixelMask[b], pr, pc)) {
                        continue;
                    }
                    if (tokenPos >= tokens[b].length || tokens[b][tokenPos] != IMG_ID) {
                        throw new FluxException("Falcon-OCR image token count is shorter than valid patches");
                    }
                    int out = 0;
                    int y0 = pr * PATCH_SIZE;
                    int x0 = pc * PATCH_SIZE;
                    for (int yy = 0; yy < PATCH_SIZE; yy++) {
                        for (int xx = 0; xx < PATCH_SIZE; xx++) {
                            float[] pixel = paddedPixels[b][y0 + yy][x0 + xx];
                            imagePatches[b][tokenPos][out++] = pixel[0];
                            imagePatches[b][tokenPos][out++] = pixel[1];
                            imagePatches[b][tokenPos][out++] = pixel[2];
                        }
                    }
                    tokenPos++;
                    written++;
                }
            }
        }
        if (written != expectedPatches) {
            throw new FluxException("Falcon-OCR patch count mismatch: " + written + " != " + expectedPatches);
        }
    }

    private static void buildPositions(long[][] tokens,
                                       long[][] posT,
                                       float[][][] posHw,
                                       boolean[][][] pixelMask,
                                       int maxHeight,
                                       int maxWidth) {
        int patchRows = maxHeight / PATCH_SIZE;
        int patchCols = maxWidth / PATCH_SIZE;
        for (int b = 0; b < tokens.length; b++) {
            int[] hw = validPatchGrid(pixelMask[b], patchRows, patchCols);
            int height = hw[0];
            int width = hw[1];
            float xlim = (float) Math.sqrt((double) width / height);
            float ylim = (float) Math.sqrt((double) height / width);
            int imageIdx = 0;
            int cumulativeTextPosition = 0;
            for (int i = 0; i < tokens[b].length; i++) {
                long token = tokens[b][i];
                if (token == IMG_ID) {
                    int row = imageIdx / width;
                    int col = imageIdx % width;
                    posHw[b][i][0] = linspace(-ylim, ylim, height, row);
                    posHw[b][i][1] = linspace(-xlim, xlim, width, col);
                    imageIdx++;
                }
                if (!isNoIncreaseToken(token)) {
                    cumulativeTextPosition++;
                }
                posT[b][i] = cumulativeTextPosition - 1L;
                if (token == PAD_TOKEN_ID) {
                    posT[b][i] = 0L;
                }
            }
        }
    }

    private static boolean[][][] buildAttentionMask(long[][] tokens) {
        int batch = tokens.length;
        int seq = tokens[0].length;
        boolean[][][] mask = new boolean[batch][seq][seq];
        for (int b = 0; b < batch; b++) {
            int firstNonPad = firstNonPadIndex(tokens[b]);
            int[] sequenceIndices = new int[seq];
            int cumulative = 0;
            for (int i = 0; i < seq; i++) {
                sequenceIndices[i] = i == 0 ? 0 : cumulative;
                boolean eos = tokens[b][i] == EOS_TOKEN_ID || i == seq - 1;
                if (eos) {
                    cumulative++;
                }
            }

            int nonPadSeen = 0;
            int[] soiCumsum = new int[seq];
            int[] eoiCumsum = new int[seq];
            int soi = 0;
            int eoi = 0;
            for (int i = 0; i < seq; i++) {
                if (tokens[b][i] != PAD_TOKEN_ID) {
                    nonPadSeen++;
                }
                if (tokens[b][i] == IMAGE_CLS_TOKEN_ID) {
                    soi++;
                }
                if (tokens[b][i] == IMG_END_ID) {
                    eoi++;
                }
                soiCumsum[i] = soi;
                eoiCumsum[i] = eoi;
                for (int kv = 0; kv <= i; kv++) {
                    boolean document = sequenceIndices[i] == sequenceIndices[kv];
                    boolean nonLeftPad = kv >= firstNonPad;
                    mask[b][i][kv] = document && nonLeftPad;
                }
            }

            for (int q = 0; q < seq; q++) {
                boolean qInImage = (soiCumsum[q] - eoiCumsum[q]) > 0;
                if (!qInImage) {
                    continue;
                }
                int qImageIndex = soiCumsum[q];
                for (int kv = 0; kv < seq; kv++) {
                    boolean kvInImage = (soiCumsum[kv] - eoiCumsum[kv]) > 0;
                    if (kvInImage && soiCumsum[kv] == qImageIndex) {
                        mask[b][q][kv] = true;
                    }
                }
            }
        }
        return mask;
    }

    private static int firstNonPadIndex(long[] tokens) {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != PAD_TOKEN_ID) {
                return i;
            }
        }
        return tokens.length;
    }

    private static boolean isNoIncreaseToken(long token) {
        return token == IMG_ID
                || token == IMAGE_REG_1_TOKEN_ID
                || token == IMAGE_REG_2_TOKEN_ID
                || token == IMAGE_REG_3_TOKEN_ID
                || token == IMAGE_REG_4_TOKEN_ID
                || token == IMG_END_ID;
    }

    private static void fillNaN(float[][] values) {
        for (float[] row : values) {
            Arrays.fill(row, Float.NaN);
        }
    }

    private static int firstImageTokenIndex(long[] tokens) {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == IMG_ID) {
                return i;
            }
        }
        throw new FluxException("No Falcon-OCR image token found");
    }

    private static int countTokens(long[][] tokens, long token) {
        int count = 0;
        for (long[] row : tokens) {
            for (long value : row) {
                if (value == token) {
                    count++;
                }
            }
        }
        return count;
    }

    private static boolean patchHasValidPixel(boolean[][] mask, int patchRow, int patchCol) {
        int y0 = patchRow * PATCH_SIZE;
        int x0 = patchCol * PATCH_SIZE;
        for (int y = 0; y < PATCH_SIZE; y++) {
            for (int x = 0; x < PATCH_SIZE; x++) {
                if (mask[y0 + y][x0 + x]) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int[] validPatchGrid(boolean[][] mask, int patchRows, int patchCols) {
        int height = 0;
        int width = 0;
        for (int pr = 0; pr < patchRows; pr++) {
            int rowWidth = 0;
            for (int pc = 0; pc < patchCols; pc++) {
                if (patchHasValidPixel(mask, pr, pc)) {
                    rowWidth++;
                }
            }
            if (rowWidth > 0) {
                height++;
                width = Math.max(width, rowWidth);
            }
        }
        if (height == 0 || width == 0) {
            throw new FluxException("Falcon-OCR image has no valid patches");
        }
        return new int[]{height, width};
    }

    private static float linspace(float start, float end, int steps, int index) {
        if (steps <= 1) {
            return start;
        }
        return start + (end - start) * index / (steps - 1);
    }
}
