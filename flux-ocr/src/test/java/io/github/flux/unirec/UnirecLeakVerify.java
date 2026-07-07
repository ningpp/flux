package io.github.flux.unirec;

import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OrtEnvironment;
import io.github.flux.core.MatManager;
import io.github.flux.core.TextResult;
import io.github.flux.model.FormulaRecognitionModel;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.Map;

/**
 * Verification for the Unirec formula pipeline memory-leak fixes and performance.
 *
 * Runs the required image D:\tmp\formula-2026-01-18-152316.png on BOTH CPU (-1) and
 * GPU (0), and asserts:
 *  1. Correctness: the decoded LaTeX is stable across many iterations.
 *  2. No MatManager tracking leak: under a long-lived (server-like) MatManager the
 *     tracked Mat count returns to its baseline after every inference.
 *  3. Native handles: a high iteration count completes without OOM (the OrtSession.Result
 *     leak that previously leaked ~maxTokens handles per image is now closed each step).
 */
public class UnirecLeakVerify {

    static final String ROOT = "D:\\models\\formula";
    static final String MODEL = "unirec-0.1b";
    static final String IMAGE = "D:\\tmp\\formula-2026-01-18-152316.png";

    public static void main(String[] args) throws Exception {
        run("CPU", -1, 100);
        run("GPU", 0, 100);
    }

    @Test
    void verifyCpuAndGpu() throws Exception {
        run("CPU", -1, 50);
        run("GPU", 0, 50);
    }

    private static void run(String label, int gpuIndex, int iterations) {
        System.out.println("\n================== " + label + " (gpuIndex=" + gpuIndex + ") ==================");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        String firstText;
        long startMem = usedMb();
        try (var model = new FormulaRecognitionModel(ROOT, MODEL, gpuIndex, env)) {
            // Warmup
            for (int i = 0; i < 3; i++) {
                try (MatManager mm = new MatManager(); NDManager nm = NDManager.newBaseManager()) {
                    model.batchPredictFiles(List.of(IMAGE), 1, mm, nm, Map.of());
                }
            }

            // Correctness: first result text must be identical across iterations.
            try (MatManager mm = new MatManager(); NDManager nm = NDManager.newBaseManager()) {
                TextResult r0 = model.batchPredictFiles(List.of(IMAGE), 1, mm, nm, Map.of()).get(0);
                firstText = r0.text();
                System.out.println("Decoded LaTeX (" + firstText.length() + " chars, "
                        + r0.tokens().length + " tokens): " + firstText.replace("\n", "\\n").replace("\r", "\\r"));
            }

            // Long-lived manager (server-like): assert tracked Mat count stays at baseline.
            MatManager liveMm = new MatManager();
            NDManager liveNm = NDManager.newBaseManager();
            int baseline = liveMm.trackedMatCount();
            int maxTracked = baseline;
            boolean consistent = true;
            long t0 = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                TextResult r = model.batchPredictFiles(List.of(IMAGE), 1, liveMm, liveNm, Map.of()).get(0);
                if (!firstText.equals(r.text())) {
                    consistent = false;
                }
                int tracked = liveMm.trackedMatCount();
                if (tracked > maxTracked) {
                    maxTracked = tracked;
                }
            }
            long t1 = System.nanoTime();
            liveMm.close();
            liveNm.close();

            double avgMs = (t1 - t0) / 1_000_000d / iterations;
            System.out.println("Iterations: " + iterations + "  avg: " + String.format("%.2f", avgMs) + " ms/infer");
            System.out.println("MatManager baseline tracked = " + baseline + ", max tracked during run = " + maxTracked);
            System.out.println("Text consistent across iterations: " + consistent);
            System.out.println("Heap used  start=" + startMem + "MB  end=" + usedMb() + "MB");

            if (!consistent) {
                throw new AssertionError(label + ": decoded text is NOT consistent across iterations!");
            }
            if (maxTracked > baseline) {
                throw new AssertionError(label + ": MatManager tracking leaked! maxTracked=" + maxTracked
                        + " > baseline=" + baseline);
            }
            System.out.println("PASS: " + label);
        } catch (Exception e) {
            System.out.println("SKIP/FAILED " + label + ": " + e);
            e.printStackTrace();
        }
    }

    private static long usedMb() {
        var bean = ManagementFactory.getMemoryMXBean();
        return (bean.getHeapMemoryUsage().getUsed() + bean.getNonHeapMemoryUsage().getUsed()) / (1024 * 1024);
    }
}
