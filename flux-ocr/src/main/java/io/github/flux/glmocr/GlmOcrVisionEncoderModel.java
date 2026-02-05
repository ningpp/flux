/*
 *    Copyright 2026 the original author or authors.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
package io.github.flux.glmocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.github.flux.exception.FluxException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Vision encoder model for GLM-OCR.
 * 
 * GLM-OCR uses a unique vision encoder architecture that takes:
 * - pixel_values: [num_patches, 1176] - preprocessed image patches
 * - pos_ids: [num_patches, 2] - position indices for rotary embeddings
 * - max_grid_size: scalar - max(height_patches, width_patches)
 * 
 * Output:
 * - image_embeds: [num_output_tokens, 1536] - merged hidden states
 * 
 * Note: The processor handles image preprocessing and returns pixel_values
 * in the expected format. The pos_ids are computed from image_grid_thw.
 */
public class GlmOcrVisionEncoderModel implements AutoCloseable {

    private static final int SPATIAL_MERGE_SIZE = 2;
    
    private final OrtEnvironment env;
    private final OrtSession session;

    public GlmOcrVisionEncoderModel(final String modelFile,
                                    final int gpuIndex,
                                    final OrtEnvironment env) {
        try {
            this.env = env;
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (gpuIndex > -1) {
                options.addCUDA(gpuIndex);
            }
            this.session = env.createSession(modelFile, options);
        } catch (Exception e) {
            throw new FluxException(e);
        }
    }

    /**
     * Encode image patches to get vision features.
     *
     * @param pixelValues preprocessed image patches [num_patches, 1176]
     * @param imageGridThw grid info [t, h, w] - temporal, height_patches, width_patches
     * @return image embeddings [num_output_tokens, hidden_size]
     */
    public float[][] predict(float[][] pixelValues, int[] imageGridThw) throws OrtException {
        // Compute position IDs for rotary embeddings
        long[][] posIds = computePosIds(imageGridThw);
        int maxGridSize = Math.max(imageGridThw[1], imageGridThw[2]);
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        
        // pixel_values: [num_patches, 1176]
        int numPatches = pixelValues.length;
        int patchFeatures = pixelValues[0].length;
        float[] flatPixelValues = flat2d(pixelValues);
        inputs.put("pixel_values", OnnxTensor.createTensor(env, 
                FloatBuffer.wrap(flatPixelValues), 
                new long[]{numPatches, patchFeatures}));
        
        // pos_ids: [num_patches, 2]
        long[] flatPosIds = flat2dLong(posIds);
        inputs.put("pos_ids", OnnxTensor.createTensor(env,
                LongBuffer.wrap(flatPosIds),
                new long[]{posIds.length, 2}));
        
        // max_grid_size: scalar
        inputs.put("max_grid_size", OnnxTensor.createTensor(env, 
                LongBuffer.wrap(new long[]{maxGridSize}),
                new long[]{}));
        
        try (Result result = session.run(inputs)) {
            // Output: [num_output_tokens, hidden_size]
            return (float[][]) result.get(0).getValue();
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }
    }

    /**
     * Compute position IDs for rotary embeddings.
     * 
     * @param gridThw [t, h, w] - temporal, height_patches, width_patches
     * @return pos_ids [num_patches, 2] - position indices
     */
    public static long[][] computePosIds(int[] gridThw) {
        int t = gridThw[0];
        int h = gridThw[1];
        int w = gridThw[2];
        
        int hMerged = h / SPATIAL_MERGE_SIZE;
        int wMerged = w / SPATIAL_MERGE_SIZE;
        int numPatchesPerFrame = hMerged * wMerged * SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE;
        int totalPatches = t * numPatchesPerFrame;
        
        // Build position arrays
        // hpos_ids: [h, w] -> reshape to [h/2, 2, w/2, 2] -> permute(0,2,1,3) -> flatten
        // wpos_ids: [h, w] -> reshape to [h/2, 2, w/2, 2] -> permute(0,2,1,3) -> flatten
        int[] hposFlat = new int[h * w];
        int[] wposFlat = new int[h * w];
        
        // Original grid
        int[][] hpos = new int[h][w];
        int[][] wpos = new int[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                hpos[i][j] = i;
                wpos[i][j] = j;
            }
        }
        
        // Reshape and permute: [h/2, 2, w/2, 2] -> permute(0,2,1,3) -> [h/2, w/2, 2, 2]
        int idx = 0;
        for (int i0 = 0; i0 < hMerged; i0++) {
            for (int i2 = 0; i2 < wMerged; i2++) {
                for (int i1 = 0; i1 < SPATIAL_MERGE_SIZE; i1++) {
                    for (int i3 = 0; i3 < SPATIAL_MERGE_SIZE; i3++) {
                        int origH = i0 * SPATIAL_MERGE_SIZE + i1;
                        int origW = i2 * SPATIAL_MERGE_SIZE + i3;
                        hposFlat[idx] = hpos[origH][origW];
                        wposFlat[idx] = wpos[origH][origW];
                        idx++;
                    }
                }
            }
        }
        
        // Stack [hpos, wpos] and repeat t times
        long[][] posIds = new long[totalPatches][2];
        for (int frame = 0; frame < t; frame++) {
            for (int p = 0; p < numPatchesPerFrame; p++) {
                int globalIdx = frame * numPatchesPerFrame + p;
                posIds[globalIdx][0] = hposFlat[p];
                posIds[globalIdx][1] = wposFlat[p];
            }
        }
        
        return posIds;
    }

    private static float[] flat2d(float[][] arr) {
        int rows = arr.length;
        int cols = arr[0].length;
        float[] flat = new float[rows * cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[idx++] = arr[i][j];
            }
        }
        return flat;
    }

    private static long[] flat2dLong(long[][] arr) {
        int rows = arr.length;
        int cols = arr[0].length;
        long[] flat = new long[rows * cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[idx++] = arr[i][j];
            }
        }
        return flat;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
