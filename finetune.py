"""
Fine-tuning script: cahya/bert2gpt-indonesian-summarization on IndoSum
Usage: python finetune.py
"""

import json
import os
import glob
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
MODEL_NAME      = "cahya/bert2gpt-indonesian-summarization"
DATA_DIR        = "data/indosum"
OUTPUT_DIR      = "models/finetuned_summarizer"
MAX_INPUT_LEN   = 512
MAX_TARGET_LEN  = 128
BATCH_SIZE      = 4
EPOCHS          = 3
LR              = 5e-5
WARMUP_STEPS    = 200
SAVE_STEPS      = 500
EVAL_STEPS      = 500
# ──────────────────────────────────────────────────────────────────────────────


def flatten(obj):
    """Recursively flatten nested lists into a single string."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(i) for i in obj)
    return str(obj)


def load_jsonl_files(pattern):
    data = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    print(f"  Loaded {len(data)} samples from {pattern}")
    return data


class IndoSumDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input, max_target):
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target
        self.items = []
        for s in samples:
            article = flatten(s.get("paragraphs", "")).strip()
            summary = flatten(s.get("summary", "")).strip()
            if article and summary:
                self.items.append((article, summary))

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
    print("  Fine-tuning: bert2gpt on IndoSum")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU detected, training will be very slow!")
    print()

    print("Loading tokenizer and model...")
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

    print("Loading IndoSum data...")
    train_data = load_jsonl_files(os.path.join(DATA_DIR, "train.*.jsonl"))
    dev_data   = load_jsonl_files(os.path.join(DATA_DIR, "dev.*.jsonl"))

    train_dataset = IndoSumDataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    dev_dataset   = IndoSumDataset(dev_data,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
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
        logging_steps               = 100,
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
    print(f"  Output:     {OUTPUT_DIR}\n")

    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to: {OUTPUT_DIR}")
    print("\nDone! Fine-tuning complete.")


if __name__ == "__main__":
    main()