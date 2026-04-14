"""
Phase 7: Model Evaluation
Loads the fine-tuned model, evaluates ROUGE scores on the test set,
and runs predictions on the 3 required sample findings.

Usage:
    python src/modeling/evaluate.py --model_dir model \
                                     --tokenized_dir data/tokenized
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import evaluate as hf_evaluate
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Required sample findings from the project specification (Appendix)
SAMPLE_FINDINGS = [
    (
        "The trachea is midline. The cardiomediastinal silhouette is normal. "
        "The lungs are clear, without evidence of acute infiltrate or effusion. "
        "There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.",
    ),
    (
        "The lungs are clear. Heart size and mediastinal contours are normal. "
        "No osseous abnormalities.",
    ),
    (
        "AP and lateral views were obtained. Bibasilar atelectasis and small left-sided "
        "pleural effusion. Stable cardiomegaly. No pneumothorax. Mild pulmonary vascular congestion.",
    ),
]


def generate_summary(
    findings: str,
    model,
    tokenizer,
    device: torch.device,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_beams: int = 4,
) -> str:
    """Generate a summary for a single findings text."""
    inputs = tokenizer(
        findings,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_target_length,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def evaluate_model(
    model_dir: str,
    tokenized_dir: str,
    results_dir: str = "results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load model and tokenizer ---
    logger.info(f"Loading model from '{model_dir}'")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # --- Load test dataset ---
    logger.info(f"Loading test dataset from '{tokenized_dir}'")
    dataset = load_from_disk(tokenized_dir)
    test_ds = dataset["test"]
    logger.info(f"Test set size: {len(test_ds)}")

    # --- Generate predictions on test set ---
    logger.info("Generating predictions on test set (this may take several minutes)...")

    all_preds = []
    all_labels = []
    batch_size = 8

    test_ds.set_format(type="torch")

    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i : i + batch_size]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=128,
                early_stopping=True,
            )

        decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in decoded_preds])

        # Replace -100 (padding) before decoding labels
        labels_np = labels.numpy()
        labels_np = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)
        all_labels.extend([l.strip() for l in decoded_labels])

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(i + batch_size, len(test_ds))}/{len(test_ds)} examples")

    # --- Compute ROUGE scores ---
    rouge = hf_evaluate.load("rouge")
    results = rouge.compute(
        predictions=all_preds,
        references=all_labels,
        use_stemmer=True,
    )

    metrics = {
        "test_set_size": len(test_ds),
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4),
    }

    logger.info("\n" + "=" * 50)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"ROUGE-1 F1: {metrics['rouge1']:.4f}  (minimum required: 0.3000)")
    logger.info(f"ROUGE-2 F1: {metrics['rouge2']:.4f}")
    logger.info(f"ROUGE-L F1: {metrics['rougeL']:.4f}")

    if metrics["rouge1"] >= 0.3:
        logger.info("PASS: Model meets the minimum ROUGE-1 F1 threshold of 0.3")
    else:
        logger.warning("FAIL: Model does NOT meet the minimum ROUGE-1 F1 threshold of 0.3")

    # Save metrics
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(results_dir) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to '{metrics_path}'")

    # --- Sample predictions on 3 required inputs ---
    logger.info("\n" + "=" * 50)
    logger.info("SAMPLE PREDICTIONS (Required Appendix Inputs)")
    logger.info("=" * 50)

    sample_predictions = []
    for idx, (findings,) in enumerate(SAMPLE_FINDINGS, start=1):
        impression = generate_summary(findings, model, tokenizer, device)
        sample_predictions.append({"findings": findings, "impression": impression})
        logger.info(f"\nSample {idx}:")
        logger.info(f"  FINDINGS:   {findings}")
        logger.info(f"  IMPRESSION: {impression}")

    # Save sample predictions
    samples_path = Path(results_dir) / "sample_predictions.json"
    with open(samples_path, "w") as f:
        json.dump(sample_predictions, f, indent=2)
    logger.info(f"\nSaved sample predictions to '{samples_path}'")

    return metrics, sample_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned summarization model")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--tokenized_dir", type=str, default="data/tokenized")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenized_dir, args.results_dir)


if __name__ == "__main__":
    main()
