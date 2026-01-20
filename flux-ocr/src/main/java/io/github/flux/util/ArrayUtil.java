package io.github.flux.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.google.gson.Gson;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public final class ArrayUtil {

    private ArrayUtil() {
    }

    /**
     * Converts a 3D int array into a DJL NDArray of dtype INT32.
     *
     * @param manager the DJL NDManager
     * @param polys   the input array of shape [D1][D2][D3]
     * @return an NDArray of shape (D1, D2, D3) and dtype INT32
     */
    public static NDArray int3dToNDArray(NDManager manager, int[][][] polys) {
        int d1 = polys.length;
        if (d1 == 0) {
            throw new IllegalArgumentException("polys must have non-zero first dimension");
        }
        int d2 = polys[0].length;
        if (d2 == 0) {
            throw new IllegalArgumentException("polys must have non-zero second dimension");
        }
        int d3 = polys[0][0].length;
        if (d3 == 0) {
            throw new IllegalArgumentException("polys must have non-zero third dimension");
        }

        // flatten to 1D int[]
        int[] flat = new int[d1 * d2 * d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            if (polys[i].length != d2) {
                throw new IllegalArgumentException("All rows must have the same second dimension");
            }
            for (int j = 0; j < d2; j++) {
                if (polys[i][j].length != d3) {
                    throw new IllegalArgumentException("All rows must have the same third dimension");
                }
                for (int k = 0; k < d3; k++) {
                    flat[idx++] = polys[i][j][k];
                }
            }
        }

        // create INT32 NDArray with shape (d1, d2, d3)
        return manager.create(flat, new Shape(d1, d2, d3));
    }

    public static boolean allTrue(boolean[] flags) {
        for (boolean flag : flags) {
            if (!flag) {
                return false;
            }
        }
        return true;
    }

    /**
     * 对一维 float 数组执行 softmax
     * @param input 输入 float[]
     * @return softmax 后的 float[]，和为 1
     */
    public static float[] softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : input) {
            if (v > max) max = v;
        }

        // 计算 exp(x - max) 并求和
        double sum = 0.0;
        double[] exp = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            exp[i] = Math.exp(input[i] - max);
            sum += exp[i];
        }

        // 归一化
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float)(exp[i] / sum);
        }

        return output;
    }

    /**
     * 等价于 PyTorch: logits.softmax(dim=1)
     *
     * @param logits shape: [batch, numClasses]
     * @return probs shape: [batch, numClasses]
     */
    public static float[][] softmaxDim1(float[][] logits) {
        int batch = logits.length;
        int numClasses = logits[0].length;

        float[][] probs = new float[batch][numClasses];

        for (int i = 0; i < batch; i++) {
            // 1. 找 max（数值稳定）
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < numClasses; j++) {
                if (logits[i][j] > max) {
                    max = logits[i][j];
                }
            }

            // 2. exp(x - max) 并求和
            float sum = 0.0f;
            for (int j = 0; j < numClasses; j++) {
                probs[i][j] = (float) Math.exp(logits[i][j] - max);
                sum += probs[i][j];
            }

            // 3. 归一化
            for (int j = 0; j < numClasses; j++) {
                probs[i][j] /= sum;
            }
        }

        return probs;
    }

    public static float max(float[] row) {
        float max = Float.MIN_VALUE; // for numerical stability
        for (float f : row) {
            if (f > max) {
                max = f;
            }
        }
        return max;
    }

    public static long argmax(float[] arr) {
        long idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    /**
     * Convert NDArray to float[][] directly
     */
    public static float[][] convertToFloatArray(NDArray ndArray) {
        // Get the shape of the NDArray
        Shape shape = ndArray.getShape();

        // Ensure it's 2D
        if (shape.dimension() != 2) {
            throw new IllegalArgumentException("NDArray must be 2D for long[][] conversion");
        }

        // Convert to float array (flattened)
        float[] flatArray = ndArray.toFloatArray();

        // Reshape to 2D
        int rows = (int) shape.get(0);
        int cols = (int) shape.get(1);

        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flatArray, i * cols, result[i], 0, cols);
        }

        return result;
    }

    /**
     * Convert NDArray to long[][] directly
     */
    public static long[][] convertToLongArray(NDArray ndArray) {
        // Get the shape of the NDArray
        Shape shape = ndArray.getShape();

        // Ensure it's 2D
        if (shape.dimension() != 2) {
            throw new IllegalArgumentException("NDArray must be 2D for long[][] conversion");
        }

        // Convert to long array (flattened)
        long[] flatArray = ndArray.toLongArray();

        // Reshape to 2D
        int rows = (int) shape.get(0);
        int cols = (int) shape.get(1);

        long[][] result = new long[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flatArray, i * cols, result[i], 0, cols);
        }

        return result;
    }

    public static <E> List<List<E>> splitIntoBatches(List<E> list, int batchSize) {
        List<List<E>> batches = new ArrayList<>();
        int totalSize = list.size();
        for (int i = 0; i < totalSize; i += batchSize) {
            int end = Math.min(i + batchSize, totalSize);
            batches.add(list.subList(i, end));
        }
        return batches;
    }

    public static boolean contains(long[] nextTokenIds, long lvalue) {
        if (nextTokenIds == null || nextTokenIds.length == 0) {
            return false;
        }
        int length = nextTokenIds.length;
        for (int i = 0; i < length; i++) {
            if (nextTokenIds[i] == lvalue) {
                return true;
            }
        }
        return false;
    }


    public static long[] clone(long[] array1) {
        if (array1 == null) {
            return null;
        }
        return array1.clone();
    }

    /** 拼接二维数组 (curInputIds + nextIds) */
    public static long[][] concat(long[][] a, long[][] b) {
        int batch = a.length;
        long[][] out = new long[batch][];
        for (int i = 0; i < batch; i++) {
            out[i] = concat(a[i], b[i]);
        }
        return out;
    }

    /**
     * Concatenates two long arrays into a single array.
     *
     * @param array1 the first array
     * @param array2 the second array
     * @return a new array containing all elements from array1 followed by all elements from array2
     */
    public static long[] concat(long[] array1, long[] array2) {
        if (array1 == null && array2 == null) {
            return new long[0];
        }
        if (array1 == null) {
            return array2.clone();
        }
        if (array2 == null) {
            return array1.clone();
        }

        long[] result = new long[array1.length + array2.length];
        System.arraycopy(array1, 0, result, 0, array1.length);
        System.arraycopy(array2, 0, result, array1.length, array2.length);
        return result;
    }

    public static OnnxTensor createOnnxTensor(float[][][][][] data, OrtEnvironment env) throws OrtException {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;
        int d5 = data[0][0][0][0].length;
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat(data)), new long[] {d1, d2, d3, d4, d5});
    }

    public static float[] flat(float[][][][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;
        int d5 = data[0][0][0][0].length;
        float[] flat = new float[d1 * d2 * d3 * d4 * d5];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    for (int l = 0; l < d4; l++) {
                        for (int m = 0; m < d5; m++) {
                            flat[idx++] = data[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        // 创建并 reshape
        return flat;
    }

    public static float[][][][] createZeros(int i, int j, int k, int m) {
        return new float[Math.max(i, 1)][Math.max(j, 1)][Math.max(k, 1)][Math.max(m, 1)];
    }

    public static OnnxTensor createOnnxTensor(float[][][][] data, OrtEnvironment env) throws OrtException {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(flat(data)), new long[] {d1, d2, d3, d4});
    }

    public static float[][][] reshape(float[] flat, int d1, int d2, int d3) {
        float[][][] data = new float[d1][d2][d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    data[i][j][k] = flat[idx++];
                }
            }
        }
        return data;
    }

    public static float[] flat(float[][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        float[] flat = new float[d1 * d2 * d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    flat[idx++] = data[i][j][k];
                }
            }
        }
        // 创建并 reshape
        return flat;
    }

    public static float[] flat(float[][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        float[] flat = new float[d1 * d2];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                flat[idx++] = data[i][j];
            }
        }
        // 创建并 reshape
        return flat;
    }

    public static NDArray toNDArray(NDManager manager, float[][][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;
        // 创建并 reshape
        return manager.create(flat(data), new Shape(d1, d2, d3, d4));
    }

    public static float[] flat(float[][][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;
        float[] flat = new float[d1 * d2 * d3 * d4];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    for (int l = 0; l < d4; l++) {
                        flat[idx++] = data[i][j][k][l];
                    }
                }
            }
        }
        // 创建并 reshape
        return flat;
    }

    public static NDArray toNDArray(NDManager manager, long[][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        // 创建并 reshape
        return manager.create(flat(data), new Shape(d1, d2));
    }

    public static long[] flat(long[][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        long[] flat = new long[d1 * d2];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                flat[idx++] = data[i][j];
            }
        }
        // 创建并 reshape
        return flat;
    }

    public static NDArray toNDArray(NDManager manager, float[][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        // 创建并 reshape
        return manager.create(flat(data), new Shape(d1, d2, d3));
    }

    public static int[] convertToInts(float[] array) {
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = Float.valueOf(array[i]).intValue();
        }
        return result;
    }

    public static int[][] convertToInts(long[][] array) {
        int[][] result = new int[array.length][];
        for (int i = 0; i < array.length; i++) {
            result[i] = new int[array[i].length];
            for (int j = 0; j < array[i].length; j++) {
                long value = array[i][j];
                if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
                    throw new ArithmeticException("long value " + value + " exceeds int range");
                }
                result[i][j] = (int) value;
            }
        }
        return result;
    }

    public static String toString(int[][][] datas) {
        if (datas == null) {
            return null;
        }
        return new Gson().toJson(datas);
    }

}
