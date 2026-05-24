import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, EncoderDecoderModel
from rouge_score import rouge_scorer
from tqdm import tqdm
import re

# CONFIG
MODEL_DIR    = "cahya/bert2gpt-indonesian-summarization"
DATA_DIR     = "data/liputan6/liputan6_data/canonical"
MAX_INPUT    = 512
MAX_TARGET   = 128
NUM_BEAMS    = 4
MAX_SAMPLES  = 200


def flatten(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(i) for i in obj)
    return str(obj)


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def lead2(text):
    """Return first 2 sentences (Liputan6 baseline is LEAD-2)."""
    sentences = split_sentences(text)
    return " ".join(sentences[:2])


def load_test_data(folder, max_samples):
    data = []
    files = sorted(Path(folder).glob("*.json"))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            article = flatten(s.get("clean_article", "")).strip()
            summary = flatten(s.get("clean_summary", "")).strip()
            if article and summary:
                data.append((article, summary))
        except Exception:
            continue
        if len(data) >= max_samples:
            break
    return data


def main():
    print("=" * 60)
    print("  ROUGE Evaluation on Liputan6 Test Set")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    # Model evaluation
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = EncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    print("  Model loaded!\n")

    print(f"Loading test data (max {MAX_SAMPLES} samples)...")
    test_data = load_test_data(os.path.join(DATA_DIR, "test"), MAX_SAMPLES)
    print(f"  Loaded {len(test_data)} samples\n")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    predictions, references = [], []

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

    # Model ROUGE
    r1_list, r2_list, rl_list = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

    model_r1 = sum(r1_list) / len(r1_list) * 100
    model_r2 = sum(r2_list) / len(r2_list) * 100
    model_rl = sum(rl_list) / len(rl_list) * 100

    # LEAD-2 Baseline
    bl_r1, bl_r2, bl_rl = [], [], []
    for article, reference in test_data:
        pred = lead2(article)
        scores = scorer.score(reference, pred)
        bl_r1.append(scores["rouge1"].fmeasure)
        bl_r2.append(scores["rouge2"].fmeasure)
        bl_rl.append(scores["rougeL"].fmeasure)

    lead2_r1 = sum(bl_r1) / len(bl_r1) * 100
    lead2_r2 = sum(bl_r2) / len(bl_r2) * 100
    lead2_rl = sum(bl_rl) / len(bl_rl) * 100

    # Print Results
    print("\n" + "=" * 50)
    print("  HASIL EVALUASI ROUGE - LIPUTAN6")
    print("=" * 50)
    print(f"  {'Method':<25} {'R1':>6} {'R2':>6} {'RL':>6}")
    print(f"  {'-'*45}")
    print(f"  {'LEAD-2 (Baseline)':<25} {lead2_r1:>6.2f} {lead2_r2:>6.2f} {lead2_rl:>6.2f}")
    print(f"  {'BERT2GPT (no fine-tune)':<25} {model_r1:>6.2f} {model_r2:>6.2f} {model_rl:>6.2f}")
    print("=" * 50)
    print(f"\n  Evaluated on {len(test_data)} test samples")

    # Save results
    results = {
        "model": MODEL_DIR,
        "num_samples": len(test_data),
        "lead2_baseline": {"rouge1": round(lead2_r1, 2), "rouge2": round(lead2_r2, 2), "rougeL": round(lead2_rl, 2)},
        "finetuned_liputan6": {"rouge1": round(model_r1, 2), "rouge2": round(model_r2, 2), "rougeL": round(model_rl, 2)},
    }
    with open("liputan6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to liputan6_results.json")

    # Show examples
    print("\n--- Contoh Prediksi ---")
    for i in range(min(3, len(predictions))):
        print(f"\n[{i+1}] Referensi : {references[i][:150]}...")
        print(f"    Prediksi  : {predictions[i][:150]}...")


if __name__ == "__main__":
    main()
