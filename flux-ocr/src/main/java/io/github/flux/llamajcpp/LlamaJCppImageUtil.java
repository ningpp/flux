package io.github.flux.llamajcpp;

import io.github.flux.exception.FluxException;
import org.opencv.core.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

final class LlamaJCppImageUtil {

    private LlamaJCppImageUtil() {
    }

    static BufferedImage toBufferedImage(final Mat rgbMat) {
        if (rgbMat == null || rgbMat.empty()) {
            throw new FluxException("Input image is empty");
        }
        if (rgbMat.channels() == 1) {
            return grayscale(rgbMat);
        }
        if (rgbMat.channels() != 3) {
            throw new FluxException("Unsupported image channels for llamaj.cpp: " + rgbMat.channels());
        }
        final BufferedImage image = new BufferedImage(rgbMat.cols(), rgbMat.rows(), BufferedImage.TYPE_INT_RGB);
        final byte[] data = new byte[rgbMat.rows() * rgbMat.cols() * rgbMat.channels()];
        rgbMat.get(0, 0, data);
        int offset = 0;
        for (int y = 0; y < rgbMat.rows(); y++) {
            for (int x = 0; x < rgbMat.cols(); x++) {
                final int red = Byte.toUnsignedInt(data[offset++]);
                final int green = Byte.toUnsignedInt(data[offset++]);
                final int blue = Byte.toUnsignedInt(data[offset++]);
                image.setRGB(x, y, (red << 16) | (green << 8) | blue);
            }
        }
        return image;
    }

    static byte[] toPngBytes(final Mat rgbMat) {
        try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            if (!ImageIO.write(toBufferedImage(rgbMat), "png", outputStream)) {
                throw new FluxException("Failed to encode image as PNG for llamaj.cpp");
            }
            return outputStream.toByteArray();
        } catch (IOException e) {
            throw new FluxException("Failed to encode image for llamaj.cpp", e);
        }
    }

    private static BufferedImage grayscale(final Mat mat) {
        final BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), BufferedImage.TYPE_BYTE_GRAY);
        final byte[] data = new byte[mat.rows() * mat.cols()];
        mat.get(0, 0, data);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
        return image;
    }
}
