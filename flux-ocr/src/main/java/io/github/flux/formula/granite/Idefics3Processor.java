// this code is convert from https://github.com/huggingface/transformers
// transformers's source code IS Licensed under the Apache License Version 2.0
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
package io.github.flux.formula.granite;

import java.util.Locale;

public class Idefics3Processor {

    public static String apply_chat_template(String message) {
        String chat_template = "<|start_of_role|>user<|end_of_role|><image>%s<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        return String.format(Locale.ROOT, chat_template, message);
    }

}
