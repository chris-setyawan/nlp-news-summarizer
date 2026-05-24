"""
Fine-tuning script: cahya/bert2gpt-indonesian-summarization on Liputan6
Usage: python finetune_liputan6.py
"""

import json
import os
import glob
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME      = "models/finetuned_summarizer"   # lanjut dari model IndoSum
DATA_DIR        = "data/liputan6/liputan6_data/canonical"
OUTPUT_DIR      = "models/finetuned_liputan6"
MAX_INPUT_LEN   = 512
MAX_TARGET_LEN  = 128
BATCH_SIZE      = 4
EPOCHS          = 3
LR              = 3e-5        # lebih kecil karena fine-tune dari model yang sudah ada
WARMUP_STEPS    = 500
SAVE_STEPS      = 1000
EVAL_STEPS      = 1000
MAX_TRAIN       = 50000       # pakai 50K dari 193K supaya tidak terlalu lama
MAX_DEV         = 2000
# ──────────────────────────────────────────────────────────────────────────────


def flatten(obj):
    """Recursively flatten nested lists of tokens into a single string."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(i) for i in obj)
    return str(obj)


def load_liputan6_folder(folder, max_samples):
    """Load individual JSON files from a Liputan6 folder."""
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
    print(f"  Loaded {len(data)} samples from {folder}")
    return data


class SumDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input, max_target):
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target
        self.items      = samples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        article, summary = self.items[idx]

        model_inputs = self.tokenizer(
            article,
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            text_target=summary,
            max_length=self.max_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


def main():
    print("=" * 60)
    print("  Fine-tuning: bert2gpt on Liputan6")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU detected, training will be very slow!")
    print()

    print(f"Loading model from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = EncoderDecoderModel.from_pretrained(MODEL_NAME)

    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
    pad_id = tokenizer.pad_token_id or 0

    model.config.decoder_start_token_id         = bos_id
    model.config.eos_token_id                   = eos_id
    model.config.pad_token_id                   = pad_id
    model.config.max_length                     = MAX_TARGET_LEN
    model.config.no_repeat_ngram_size           = 3
    model.config.early_stopping                 = True
    model.config.num_beams                      = 4
    model.decoder.config.decoder_start_token_id = bos_id
    model.decoder.config.bos_token_id           = bos_id
    print("  Model loaded!\n")

    print("Loading Liputan6 data...")
    train_data = load_liputan6_folder(os.path.join(DATA_DIR, "train"), MAX_TRAIN)
    dev_data   = load_liputan6_folder(os.path.join(DATA_DIR, "dev"),   MAX_DEV)

    train_dataset = SumDataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    dev_dataset   = SumDataset(dev_data,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Dev:   {len(dev_dataset)} samples\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LR,
        weight_decay                = 0.01,
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = 200,
        evaluation_strategy         = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        predict_with_generate       = True,
        fp16                        = True,
        report_to                   = "none",
        dataloader_num_workers      = 2,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = dev_dataset,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR:         {LR}")
    print(f"  Train:      {len(train_dataset)} samples")
    print(f"  Output:     {OUTPUT_DIR}\n")

    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to: {OUTPUT_DIR}")
    print("\nDone! Fine-tuning Liputan6 complete.")


if __name__ == "__main__":
    main()
