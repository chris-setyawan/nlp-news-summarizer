from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import json
import torch

# CONFIG
MODEL_NAME  = "cahya/bert-base-indonesian-NER"
MAX_SAMPLES = 200

# Map label NERGrit → label standar
NERGRIT_MAP = {
    "B-PERSON":       "B-PER",
    "I-PERSON":       "I-PER",
    "B-ORGANISATION": "B-ORG",
    "I-ORGANISATION": "I-ORG",
    "B-PLACE":        "B-LOC",
    "I-PLACE":        "I-LOC",
    "O":              "O",
}

# Map label output model → label standar
MODEL_LABEL_MAP = {
    "B-PER": "B-PER", "I-PER": "I-PER",
    "B-ORG": "B-ORG", "I-ORG": "I-ORG",
    "B-LOC": "B-LOC", "I-LOC": "I-LOC",
    "B-GPE": "B-LOC", "I-GPE": "I-LOC",  # GPE → LOC
    "O": "O",
}


def align_predictions(tokens, preds):
    pred_labels = ["O"] * len(tokens)

    # Rekonstruksi posisi karakter per token
    char_positions = []
    pos = 0
    for token in tokens:
        char_positions.append((pos, pos + len(token)))
        pos += len(token) + 1  # +1 untuk spasi

    for pred in preds:
        raw = pred["entity"]
        # Normalize label
        label = MODEL_LABEL_MAP.get(raw, "O")
        if label == "O":
            continue

        p_start = pred["start"]
        p_end   = pred["end"]

        first = True
        for i, (t_start, t_end) in enumerate(char_positions):
            # Cek overlap
            if t_start < p_end and t_end > p_start:
                if first:
                    # Gunakan B- untuk token pertama
                    pred_labels[i] = "B-" + label.split("-")[1]
                    first = False
                else:
                    pred_labels[i] = "I-" + label.split("-")[1]

    return pred_labels


def main():
    print("=" * 60)
    print("  NER Evaluation on IndoNLU NERGrit Test Set")
    print("=" * 60)

    # Load dataset
    print("\nLoading NERGrit dataset...")
    dataset    = load_dataset("indonlp/indonlu", "nergrit", trust_remote_code=True)
    test_data  = dataset["test"]
    label_names = test_data.features["ner_tags"].feature.names
    print(f"  Label names  : {label_names}")
    print(f"  Test samples : {len(test_data)}")

    # Load model (token-level, aggregation_strategy=none)
    print(f"\nLoading NER model '{MODEL_NAME}'...")
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        task="ner",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        aggregation_strategy="none",
        device=device,
    )
    print("  Model loaded!\n")

    all_true  = []
    all_pred  = []
    samples   = test_data.select(range(min(MAX_SAMPLES, len(test_data))))

    print(f"Evaluating on {len(samples)} samples...")
    for sample in tqdm(samples):
        tokens      = sample["tokens"]
        true_ids    = sample["ner_tags"]
        true_labels = [NERGRIT_MAP.get(label_names[t], "O") for t in true_ids]

        sentence = " ".join(tokens)

        try:
            preds       = ner_pipeline(sentence)
            pred_labels = align_predictions(tokens, preds)
        except Exception:
            pred_labels = ["O"] * len(tokens)

        all_true.append(true_labels)
        all_pred.append(pred_labels)

    # Compute metrics
    print("\n" + "=" * 60)
    print("  HASIL EVALUASI NER - IndoBERT-NER")
    print("=" * 60)
    report = classification_report(all_true, all_pred, digits=4)
    print(report)

    p  = precision_score(all_true, all_pred)
    r  = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)

    print(f"  Precision (micro): {p:.4f}")
    print(f"  Recall    (micro): {r:.4f}")
    print(f"  F1-Score  (micro): {f1:.4f}")
    print("=" * 60)
    print(f"\n  Evaluated on {len(samples)} test samples")

    results = {
        "model": MODEL_NAME,
        "num_samples": len(samples),
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1_score":  round(f1, 4),
    }
    with open("ner_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to ner_results.json")


if __name__ == "__main__":
    main()
