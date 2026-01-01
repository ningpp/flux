package io.github.flux.core;

public interface Processor<I, O> {

    O process(I input);

}
