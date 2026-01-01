package io.github.flux.formula.granite;

import java.util.Locale;

public class Idefics3Processor {

    public static String apply_chat_template(String message) {
        String chat_template = "<|start_of_role|>user<|end_of_role|><image>%s<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        return String.format(Locale.ROOT, chat_template, message);
    }

}
