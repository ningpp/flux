package io.github.flux.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class CollectionUtil {

    private CollectionUtil() {
    }

    public static <E> List<List<E>> split(List<E> images, int batchSize) {
        if (images == null || images.isEmpty()) {
            return new ArrayList<>();
        }

        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be greater than 0");
        }

        List<List<E>> batches = new ArrayList<>();

        for (int i = 0; i < images.size(); i += batchSize) {
            int end = Math.min(i + batchSize, images.size());
            batches.add(new ArrayList<>(images.subList(i, end)));
        }

        return batches;
    }

    public static Set<String> distinct(Collection<String> c1, Collection<String> c2) {
        return Stream.concat(c1.stream(), c2.stream()).collect(Collectors.toSet());
    }

    public static Set<String> distinct(Collection<String> c1, Collection<String> c2, Collection<String> c3) {
        return distinct(distinct(c1, c2), c3);
    }

    public static Set<String> distinct(Collection<String> c1, Collection<String> c2, Collection<String> c3, Collection<String> c4) {
        return distinct(distinct(c1, c2, c3), c4);
    }

    public static Set<String> distinct(Collection<Collection<String>> cs) {
        return cs.stream().flatMap(Collection::stream).collect(Collectors.toSet());
    }

}
