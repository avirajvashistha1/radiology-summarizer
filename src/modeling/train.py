"""
Phase 6: Fine-Tuning BART-base for Radiology Report Summarization
Fine-tunes facebook/bart-base on the Open-i dataset using HuggingFace Seq2SeqTrainer.

Usage:
    python src/modeling/train.py --tokenized_dir data/tokenized \
                                  --output_dir model \
                                  --model_name facebook/bart-base
"""

import argparse
import importlib
import logging
import os
import random
import sys

# Prevent local evaluate.py from shadowing the 'evaluate' pip package
_orig_path = sys.path[:]
sys.path = [p for p in sys.path if os.path.basename(p) != "modeling"]
import evaluate as hf_evaluate
sys.path = _orig_path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def build_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function for Seq2SeqTrainer.
    Computes ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    """
    rouge = hf_evaluate.load("rouge")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # Replace -100 in predictions (padding) before decoding
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (padding) before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }

    return compute_metrics


def train(
    tokenized_dir: str,
    output_dir: str,
    model_name: str = "facebook/bart-base",
):
    set_all_seeds(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    use_fp16 = device.type == "cuda"

    # --- Load tokenized dataset ---
    logger.info(f"Loading tokenized dataset from '{tokenized_dir}'")
    dataset = load_from_disk(tokenized_dir)
    train_ds = dataset["train"]
    val_ds = dataset["val"]
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # --- Load tokenizer and model ---
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # --- Data collator for dynamic padding ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,         # effective batch size = 16
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=128,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,                    # keep only 2 checkpoints to save disk space
        seed=RANDOM_SEED,
        dataloader_num_workers=0,              # avoids deadlock on Windows
        report_to="none",                      # disable wandb/tensorboard by default
    )

    logger.info("Training arguments:")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  FP16: {training_args.fp16}")

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # --- Save best model and tokenizer ---
    logger.info(f"Saving best model to '{output_dir}'")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete.")
    logger.info(f"Best validation ROUGE-1: {trainer.state.best_metric:.4f}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART-base for radiology summarization")
    parser.add_argument("--tokenized_dir", type=str, default="data/tokenized")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    args = parser.parse_args()

    train(args.tokenized_dir, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
