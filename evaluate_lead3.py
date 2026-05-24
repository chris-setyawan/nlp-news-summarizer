import json
import glob
import os
import re
from rouge_score import rouge_scorer
from tqdm import tqdm

# CONFIG 
DATA_DIR     = "data/indosum"
MAX_SAMPLES  = 200


def flatten(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(i) for i in obj)
    return str(obj)


def split_sentences(text):
    """Split text into sentences by period, question mark, exclamation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def lead3(text):
    """Return first 3 sentences of text."""
    sentences = split_sentences(text)
    return " ".join(sentences[:3])


def load_test_data(pattern, max_samples):
    data = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    article = flatten(s.get("paragraphs", "")).strip()
                    summary = flatten(s.get("summary", "")).strip()
                    if article and summary:
                        data.append((article, summary))
                if len(data) >= max_samples:
                    return data
    return data


def main():
    print("=" * 60)
    print("  LEAD-3 Baseline Evaluation on IndoSum Test Set")
    print("=" * 60)

    print(f"Loading test data (max {MAX_SAMPLES} samples)...")
    test_data = load_test_data(os.path.join(DATA_DIR, "test.*.jsonl"), MAX_SAMPLES)
    print(f"  Loaded {len(test_data)} samples\n")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    r1_list, r2_list, rl_list = [], [], []

    print("Calculating LEAD-3 ROUGE scores...")
    for article, reference in tqdm(test_data):
        prediction = lead3(article)
        scores = scorer.score(reference, prediction)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

    r1 = sum(r1_list) / len(r1_list) * 100
    r2 = sum(r2_list) / len(r2_list) * 100
    rl = sum(rl_list) / len(rl_list) * 100

    print("\n" + "=" * 40)
    print("  HASIL LEAD-3 BASELINE")
    print("=" * 40)
    print(f"  ROUGE-1 : {r1:.2f}")
    print(f"  ROUGE-2 : {r2:.2f}")
    print(f"  ROUGE-L : {rl:.2f}")
    print("=" * 40)
    print(f"\n  Evaluated on {len(test_data)} test samples")

    # Save results
    results = {
        "method": "LEAD-3",
        "num_samples": len(test_data),
        "rouge1": round(r1, 2),
        "rouge2": round(r2, 2),
        "rougeL": round(rl, 2),
    }
    with open("lead3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to lead3_results.json")

    # Show 3 examples
    print("\n--- Contoh LEAD-3 ---")
    for i, (article, reference) in enumerate(test_data[:3]):
        pred = lead3(article)
        print(f"\n[{i+1}] Referensi : {reference[:150]}...")
        print(f"    LEAD-3    : {pred[:150]}...")


if __name__ == "__main__":
    main()
