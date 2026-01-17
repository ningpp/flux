package io.github.flux.formula.texteller;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.flux.core.FormulaRecognitionResult;
import io.github.flux.exception.FluxException;
import io.github.flux.util.ArrayUtil;
import io.github.flux.util.IOUtil;
import io.github.flux.util.OnnxUtil;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TexTellerDecoderModel implements AutoCloseable {

    @Override
    public void close() throws Exception {
        session.close();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int maxLength;
    private final long padTokenId;
    private final long eosTokenId;
    private final long decoderStartTokenId;
    private final HuggingFaceTokenizer tokenizer;


    public TexTellerDecoderModel(final String modelFile,
                                 final int gpuIndex,
                                 final OrtEnvironment env,
                                 int maxLength,
                                 long padTokenId,
                                 long eosTokenId,
                                 long decoderStartTokenId,
                                 HuggingFaceTokenizer tokenizer) {
        this.maxLength = maxLength;
        this.eosTokenId = eosTokenId;
        this.padTokenId = padTokenId;
        this.tokenizer = tokenizer;
        this.decoderStartTokenId = decoderStartTokenId;
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

    public List<FormulaRecognitionResult> batchPredict(float[][][] encodeResultFloats, NDManager ndManager) throws OrtException {
        int layers = 12;
        int num_attention_heads = 16;
        int embed_size_per_head = 64;
        NDArray inputNdArray = ArrayUtil.toNDArray(ndManager, encodeResultFloats);
        FloatBuffer dataBuffer = inputNdArray.toByteBuffer().asFloatBuffer();
        long[] shape = inputNdArray.getShape().getShape();
        int batchSize = (int) shape[0];
        OnnxTensor encoderHiddenStates = OnnxTensor.createTensor(env, dataBuffer, shape);
        List<List<Long>> allGeneratedIds = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            allGeneratedIds.add(new ArrayList<>(List.of(decoderStartTokenId)));
        }
        boolean[] finished = new boolean[batchSize];

        List<Map<String, OnnxTensor>> past_kvs = new ArrayList<>();
        List<Map<String, OnnxTensor>> cross_attention_kvs = new ArrayList<>();
        Map<String, OnnxTensor> emptyPastKV = new HashMap<>();
        for (int step = 0; step < maxLength; step++) {
            Map<String, OnnxTensor> cross_attention_key_values = new HashMap<>();
            Map<String, OnnxTensor> onnxInputs = new LinkedHashMap<>();

            long[][] inputIdsArray;

            if (step == 0) {
                inputIdsArray = new long[batchSize][];
                for (int i = 0; i < batchSize; i++) {
                    inputIdsArray[i] = new long[]{decoderStartTokenId};
                }
                onnxInputs.put("use_cache_branch",
                        OnnxTensor.createTensor(env, new boolean[]{false}));
                for (int i = 0; i < layers; i++) {
                    emptyPastKV.put("past_key_values." + i + ".decoder.key",
                            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[]{batchSize, num_attention_heads, 0, embed_size_per_head}));
                    emptyPastKV.put("past_key_values." + i + ".decoder.value",
                            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[]{batchSize, num_attention_heads, 0, embed_size_per_head}));
                    emptyPastKV.put("past_key_values." + i + ".encoder.key",
                            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[]{batchSize, num_attention_heads, 0, embed_size_per_head}));
                    emptyPastKV.put("past_key_values." + i + ".encoder.value",
                            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), new long[]{batchSize, num_attention_heads, 0, embed_size_per_head}));
                }
                onnxInputs.putAll(emptyPastKV);
            } else {
                cross_attention_kvs.add(cross_attention_key_values);
                for (int i = 0; i < layers; i++) {
                    String keyKey = "past_key_values." + i + ".encoder.key";
                    cross_attention_key_values.put(keyKey, past_kvs.get(step-1).get(keyKey));
                    String valKey = "past_key_values." + i + ".encoder.value";
                    cross_attention_key_values.put(valKey, past_kvs.get(step-1).get(valKey));
                }
                // 后续推理：只传入最新生成的 token
                inputIdsArray = new long[batchSize][];
                for (int i = 0; i < batchSize; i++) {
                    inputIdsArray[i] = new long[]{allGeneratedIds.get(i).getLast()};
                }
                onnxInputs.put("use_cache_branch",
                        OnnxTensor.createTensor(env, new boolean[]{true}));

                // 添加上一次输出的 past_key_values
                onnxInputs.putAll(past_kvs.get(step-1));
            }

            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsArray);
            onnxInputs.put("input_ids", inputIdsTensor);
            onnxInputs.put("encoder_hidden_states", encoderHiddenStates);

            OrtSession.Result result = session.run(onnxInputs);
            OnnxUtil.closeTensors(emptyPastKV);

            Map<String, OnnxTensor> pastKeyValues = new HashMap<>();
            past_kvs.add(pastKeyValues);
            for (int i = 0; i < layers; i++) {
                String keyKeyEncoder = "past_key_values." + i + ".encoder.key";
                String valKeyEncoder = "past_key_values." + i + ".encoder.value";
                if (step == 0) {
                    pastKeyValues.put(keyKeyEncoder, (OnnxTensor) result.get("present." + i + ".encoder.key").get());
                    pastKeyValues.put(valKeyEncoder, (OnnxTensor) result.get("present." + i + ".encoder.value").get());
                } else {
                    pastKeyValues.put(keyKeyEncoder, cross_attention_key_values.get(keyKeyEncoder));
                    pastKeyValues.put(valKeyEncoder, cross_attention_key_values.get(valKeyEncoder));
                }

                String keyKeyDecoder = "past_key_values." + i + ".decoder.key";
                String valKeyDecoder = "past_key_values." + i + ".decoder.value";
                pastKeyValues.put(keyKeyDecoder, (OnnxTensor) result.get("present." + i + ".decoder.key").get());
                pastKeyValues.put(valKeyDecoder, (OnnxTensor) result.get("present." + i + ".decoder.value").get());
            }

            // 获取 logits 并选择下一个 token
            OnnxTensor logits = (OnnxTensor) result.get("logits").get();
            long[] nextTokens = selectNextTokens(logits);

            for (int b = 0; b < batchSize; b++) {
                if (!finished[b]) {
                    allGeneratedIds.get(b).add(nextTokens[b]);

                    // 检查是否遇到 EOS 或达到最大长度
                    if (nextTokens[b] == eosTokenId) {
                        finished[b] = true;
                    }
                }
            }
            IOUtil.close(logits);

            // 清理当前步骤的输入 tensors
            inputIdsTensor.close();
            onnxInputs.get("use_cache_branch").close();

            // 检查是否生成结束
            if (ArrayUtil.allTrue(finished)) {
                break;
            }
        }

        past_kvs.forEach(OnnxUtil::closeTensors);
        IOUtil.close(encoderHiddenStates);
        OnnxUtil.closeTensors(emptyPastKV);
        cross_attention_kvs.forEach(OnnxUtil::closeTensors);

        List<FormulaRecognitionResult> results = new ArrayList<>();
        for (var generatedIds : allGeneratedIds) {
            long[] tokens = generatedIds.stream().mapToLong(Long::longValue).toArray();
            String text = tokenizer.decode(tokens, true);
            results.add(new FormulaRecognitionResult(List.of(addNewlines(removeStyle(text))), tokens, -1));
        }
        return results;
    }

    private static String removeStyle(String inputStr) {
        if (inputStr == null) return null;

        // 定义需要匹配的关键字，对应原 Python 代码中的各指令
        String[] keywords = {"bm", "boldsymbol", "textit", "textbf", "mathbf"};

        String result = inputStr;
        for (String keyword : keywords) {
            // 正则解释: \\ keyword \{ (内容) \}
            // 使用非贪婪匹配 (.*?) 以确保正确处理多个标签
            String regex = "\\\\" + keyword + "\\s*\\{(.*?)\\}";
            Pattern pattern = Pattern.compile(regex);
            Matcher matcher = pattern.matcher(result);

            // 将 \cmd{text} 替换为 text
            result = matcher.replaceAll("$1");
        }

        result = result.trim();
        if (result.startsWith("\\[") && result.endsWith("\\]")) {
            // remove for formula
            result = result.substring(2, result.length()-2);
        }
        return result;
    }

    /**
     * 为 LaTeX 字符串添加换行符，确保 begin/end 环境周围没有重复换行。
     *
     * @param latexStr 输入的 LaTeX 字符串
     * @return 处理后的 LaTeX 字符串
     */
    public static String addNewlines(String latexStr) {
        if (latexStr == null) return "";

        String processedStr = latexStr;

        // 1. 在 \begin{...} 前后添加换行，并替换周围原有的空白字符
        // Java 中 \\\\ 代表正则中的一个反斜杠，[^}]* 匹配非右花括号的内容
        // $1 代表第一个捕获组 (\\begin\{[^}]*\})
        processedStr = processedStr.replaceAll("\\s*(\\\\begin\\{[^}]*\\})\\s*", "\n$1\n");

        // 2. 在 \end{...} 前后添加换行
        processedStr = processedStr.replaceAll("\\s*(\\\\end\\{[^}]*\\})\\s*", "\n$1\n");

        // 3. 在 \\ 后面添加换行（如果后面没有紧跟换行或空格）
        // (?!...) 是负向先行断言
        // 正则含义：匹配 \\ 且后面不跟 \n 或空格，或者匹配 \\ 后接一个空格
        processedStr = processedStr.replaceAll("\\\\\\\\(?!\\n| )|\\\\\\\\ ", "\\\\\\\\\n");

        // 4. 清理：将连续的两个或多个换行符压缩为一个换行符
        processedStr = processedStr.replaceAll("\\n{2,}", "\n");

        // 去除首尾空白字符（类似于 Python 的 strip()）
        return processedStr.trim();
    }

    private long[] selectNextTokens(OnnxTensor logits) throws OrtException {
        // logits 维度通常为: [batch_size, seq_len, vocab_size]
        float[][][] logitsData = (float[][][]) logits.getValue();
        int batchSize = logitsData.length;
        int seqLen = logitsData[0].length;
        long[] nextTokens = new long[batchSize];

        for (int b = 0; b < batchSize; b++) {
            // 取当前 batch 中最后一个时间步的 logits
            float[] lastLogits = logitsData[b][seqLen - 1];

            // Greedy Search
            int maxIdx = 0;
            float maxVal = lastLogits[0];
            for (int i = 1; i < lastLogits.length; i++) {
                if (lastLogits[i] > maxVal) {
                    maxVal = lastLogits[i];
                    maxIdx = i;
                }
            }
            nextTokens[b] = maxIdx;
        }
        return nextTokens;
    }

    /**
     * 从 logits 中选择下一个 token (这里用 greedy search)
     */
    private long selectNextToken(OnnxTensor logits) throws OrtException {
        // logits shape: [batch_size, seq_len, vocab_size]
        float[][][] logitsData = (float[][][]) logits.getValue();

        // 取最后一个位置的 logits
        float[] lastLogits = logitsData[0][logitsData[0].length - 1];

        // Greedy:  选择概率最大的 token
        int maxIdx = 0;
        float maxVal = lastLogits[0];
        for (int i = 1; i < lastLogits.length; i++) {
            if (lastLogits[i] > maxVal) {
                maxVal = lastLogits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

}
