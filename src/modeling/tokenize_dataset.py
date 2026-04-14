"""
Phase 4: Tokenization and Dataset Preparation
Loads train/val/test CSVs, tokenizes with BART tokenizer,
and saves HuggingFace Dataset objects to disk.

Usage:
    python src/modeling/tokenize_dataset.py --data_dir data/cleaned \
                                             --output_dir data/tokenized \
                                             --model_name facebook/bart-base
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128


def build_tokenize_fn(tokenizer, max_source_length: int, max_target_length: int):
    """
    Returns a tokenization function suitable for dataset.map().
    Replaces padding token IDs in labels with -100 so the loss ignores padding.
    """

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["findings"],
            max_length=max_source_length,
            truncation=True,
            padding=False,  # dynamic padding handled by DataCollatorForSeq2Seq
        )

        labels = tokenizer(
            text_target=batch["impression"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

        # Replace pad token IDs with -100 to ignore padding in loss
        label_ids = []
        for ids in labels["input_ids"]:
            ids_with_ignore = [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in ids
            ]
            label_ids.append(ids_with_ignore)

        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_fn


def tokenize_splits(
    data_dir: str,
    output_dir: str,
    model_name: str = "facebook/bart-base",
) -> DatasetDict:
    """
    Load train/val/test CSVs, tokenize them, and save to disk.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datasets = {}
    for split in ["train", "val", "test"]:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {csv_path}. "
                "Run src/data/split.py first."
            )
        df = pd.read_csv(csv_path)[["findings", "impression"]]
        datasets[split] = Dataset.from_pandas(df, preserve_index=False)
        logger.info(f"Loaded {split}: {len(datasets[split])} rows")

    dataset_dict = DatasetDict(datasets)

    logger.info("Tokenizing datasets (batched)...")
    tokenize_fn = build_tokenize_fn(tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=["findings", "impression"],
        desc="Tokenizing",
    )

    logger.info(f"Saving tokenized datasets to '{output_dir}'")
    tokenized.save_to_disk(str(output_dir))

    # Also save tokenizer alongside tokenized data for convenience
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    logger.info("Tokenization complete.")
    logger.info(f"  Train features: {tokenized['train'].features}")
    logger.info(f"  Example input_ids length: {len(tokenized['train'][0]['input_ids'])}")

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Tokenize radiology report splits")
    parser.add_argument("--data_dir", type=str, default="data/cleaned")
    parser.add_argument("--output_dir", type=str, default="data/tokenized")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    args = parser.parse_args()

    tokenize_splits(args.data_dir, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
