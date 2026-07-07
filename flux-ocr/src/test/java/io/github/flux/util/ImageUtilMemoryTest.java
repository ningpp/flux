package io.github.flux.util;

import io.github.flux.core.MatManager;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ImageUtilMemoryTest {

    @BeforeAll
    static void loadOpenCv() {
        org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.opencv_java.class);
    }

    @Test
    void padImageToSizeReleasesOriginalFromMatManagerTracking() throws Exception {
        try (MatManager matManager = new MatManager()) {
            Mat original = matManager.newMat(10, 10, CvType.CV_8UC3);
            Mat padded = ImageUtil.padImageToSize(matManager, original, 20, 20, Scalar.all(255));

            matManager.release(padded);

            assertEquals(0, matManager.trackedMatCount());
        }
    }

    @Test
    void cropReturnsTrackedCloneAndReleasesTemporaryRoi() throws Exception {
        try (MatManager matManager = new MatManager()) {
            Mat original = matManager.newMat(10, 10, CvType.CV_8UC3);
            Mat cropped = ImageUtil.crop(matManager, original, 1, 1, 8, 8);

            matManager.release(cropped);
            matManager.release(original);

            assertEquals(0, matManager.trackedMatCount());
        }
    }
}
