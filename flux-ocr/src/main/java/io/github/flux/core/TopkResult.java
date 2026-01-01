package io.github.flux.core;

public record TopkResult(int[][] indices, float[][] scores, String[][] labels) {

}
