package io.github.flux.exception;

public class FluxException extends RuntimeException {

    public FluxException() {
        super();
    }

    public FluxException(String message) {
        super(message);
    }

    public FluxException(String message, Throwable cause) {
        super(message, cause);
    }

    public FluxException(Throwable cause) {
        super(cause);
    }

    protected FluxException(String message, Throwable cause,
                            boolean enableSuppression,
                            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

}
