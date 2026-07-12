import json
import pickle
from pathlib import Path

REF_PKL = Path(r"D:\code\flux\output\fp32_reference_gotocr2.pkl")
FP16_PKL = Path(r"D:\code\flux\output\fp16_results_gotocr2.pkl")
REPORT = Path(r"D:\code\flux\output\gotocr2_fp16_validation_report.json")


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def normalize(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"D:\models\onnx\GOT-OCR-2.0", trust_remote_code=True)
    ref = pickle.load(open(REF_PKL, "rb"))
    hyp = pickle.load(open(FP16_PKL, "rb"))

    per = []
    total_norm_ref = 0
    total_norm_dist = 0
    for name in sorted(ref):
        if name not in hyp:
            continue
        ref_text = tokenizer.decode(ref[name], skip_special_tokens=True)
        hyp_text = tokenizer.decode(hyp[name], skip_special_tokens=True)
        nr = normalize(ref_text)
        nh = normalize(hyp_text)
        dist = levenshtein(nr, nh)
        denom = max(len(nr), len(nh))
        acc = 1.0 if denom == 0 else 1.0 - dist / denom
        total_norm_ref += denom
        total_norm_dist += dist
        per.append({
            "image": name,
            "norm_ref_len": len(nr),
            "norm_hyp_len": len(nh),
            "norm_distance": dist,
            "norm_text_accuracy": round(acc, 6),
        })

    avg_acc = 1.0 - total_norm_dist / total_norm_ref if total_norm_ref else 1.0
    print(f"Images: {len(per)}")
    print(f"Total normalized ref chars: {total_norm_ref}")
    print(f"Total normalized edit distance: {total_norm_dist}")
    print(f"Average normalized text accuracy: {avg_acc * 100:.4f}%")
    print(f"Normalized accuracy drop: {(1 - avg_acc) * 100:.4f}%")
    print("Worst cases:")
    for p in sorted(per, key=lambda x: x["norm_text_accuracy"])[:10]:
        print(p)


if __name__ == "__main__":
    main()
