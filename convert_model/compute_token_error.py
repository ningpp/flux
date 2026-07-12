import json
import pickle
from pathlib import Path

REPORT = Path(r"D:\code\flux\output\gotocr2_fp16_validation_report.json")
REF_PKL = Path(r"D:\code\flux\output\fp32_reference_gotocr2.pkl")
FP16_PKL = Path(r"D:\code\flux\output\fp16_results_gotocr2.pkl")


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def main():
    ref = pickle.load(open(REF_PKL, "rb"))
    if FP16_PKL.exists():
        hyp = pickle.load(open(FP16_PKL, "rb"))
    else:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        # reconstruct empty hyp from lengths? not possible
        print("No FP16 pickle found; using report metadata only")
        return

    total_ref = 0
    total_dist = 0
    per = []
    for name in sorted(ref):
        r = ref[name]
        h = hyp.get(name, [])
        dist = levenshtein(r, h)
        total_ref += len(r)
        total_dist += dist
        per.append({"image": name, "ref_len": len(r), "hyp_len": len(h), "distance": dist,
                    "error_rate": dist / len(r) if r else 0.0})

    avg_error = total_dist / total_ref if total_ref else 0.0
    print(f"Total ref tokens: {total_ref}")
    print(f"Total edit distance: {total_dist}")
    print(f"Average token error rate: {avg_error * 100:.4f}%")
    print(f"Accuracy (1 - error): {(1 - avg_error) * 100:.4f}%")
    for p in sorted(per, key=lambda x: -x["error_rate"])[:10]:
        print(p)


if __name__ == "__main__":
    main()
