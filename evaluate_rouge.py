"""
ROUGE Evaluation Script
Evaluates fine-tuned summarizer on IndoSum test set
Usage: python evaluate_rouge.py
"""

import json
import glob
import os
import torch
from transformers import AutoTokenizer, EncoderDecoderModel
from rouge_score import rouge_scorer
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR    = "models/finetuned_summarizer"
DATA_DIR     = "data/indosum"
MAX_INPUT    = 512
MAX_TARGET   = 128
NUM_BEAMS    = 4
MAX_SAMPLES  = 200   # pakai 200 sample test — cukup representatif & cepat
# ──────────────────────────────────────────────────────────────────────────────


def flatten(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(i) for i in obj)
    return str(obj)


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
    print("  ROUGE Evaluation on IndoSum Test Set")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    # Load model
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = EncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    print("  Model loaded!\n")

    # Load test data
    print(f"Loading test data (max {MAX_SAMPLES} samples)...")
    test_data = load_test_data(os.path.join(DATA_DIR, "test.*.jsonl"), MAX_SAMPLES)
    print(f"  Loaded {len(test_data)} samples\n")

    # ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    predictions = []
    references  = []

    print("Generating summaries...")
    for article, reference in tqdm(test_data):
        inputs = tokenizer(
            article,
            max_length=MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_TARGET,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(reference)

    # Calculate ROUGE
    print("\nCalculating ROUGE scores...")
    r1_list, r2_list, rl_list = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

    r1 = sum(r1_list) / len(r1_list) * 100
    r2 = sum(r2_list) / len(r2_list) * 100
    rl = sum(rl_list) / len(rl_list) * 100

    print("\n" + "=" * 40)
    print("  HASIL EVALUASI ROUGE")
    print("=" * 40)
    print(f"  ROUGE-1 : {r1:.2f}")
    print(f"  ROUGE-2 : {r2:.2f}")
    print(f"  ROUGE-L : {rl:.2f}")
    print("=" * 40)
    print(f"\n  Evaluated on {len(test_data)} test samples")

    # Save results
    results = {
        "model": MODEL_DIR,
        "num_samples": len(test_data),
        "rouge1": round(r1, 2),
        "rouge2": round(r2, 2),
        "rougeL": round(rl, 2),
    }
    with open("rouge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to rouge_results.json")

    # Show 3 example predictions
    print("\n--- Contoh Prediksi ---")
    for i in range(min(3, len(predictions))):
        print(f"\n[{i+1}] Referensi : {references[i][:150]}...")
        print(f"    Prediksi  : {predictions[i][:150]}...")


if __name__ == "__main__":
    main()
