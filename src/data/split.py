"""
Phase 5: Train/Validation/Test Split
Splits the cleaned dataset into 72% train, 8% validation, 20% test.
Uses a fixed random seed for reproducibility.

Usage:
    python src/data/split.py --input_path data/cleaned/reports_clean.csv \
                              --output_dir data/cleaned
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def split_data(input_path: str, output_dir: str) -> dict:
    """
    Load cleaned CSV and produce train/val/test splits.
    Returns a dict with keys 'train', 'val', 'test' mapping to DataFrames.
    """
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from '{input_path}'")

    # First split: 80% train+val, 20% test
    train_val, test = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED)

    # Second split: 90% train, 10% val from the 80% pool → 72/8/20 overall
    train, val = train_test_split(train_val, test_size=0.10, random_state=RANDOM_SEED)

    splits = {"train": train, "val": val, "test": test}

    for name, split_df in splits.items():
        logger.info(f"{name}: {len(split_df)} rows")

    # Verify zero leakage across splits
    train_ids = set(train["report_id"])
    val_ids = set(val["report_id"])
    test_ids = set(test["report_id"])
    assert train_ids.isdisjoint(val_ids), "LEAK: train and val share report_ids"
    assert train_ids.isdisjoint(test_ids), "LEAK: train and test share report_ids"
    assert val_ids.isdisjoint(test_ids), "LEAK: val and test share report_ids"
    logger.info("Split integrity check passed — no leakage between splits")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        path = output_dir / f"{name}.csv"
        split_df.reset_index(drop=True).to_csv(path, index=False)
        logger.info(f"Saved '{path}'")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Split cleaned reports into train/val/test")
    parser.add_argument("--input_path", type=str, default="data/cleaned/reports_clean.csv")
    parser.add_argument("--output_dir", type=str, default="data/cleaned")
    args = parser.parse_args()

    splits = split_data(args.input_path, args.output_dir)
    for name, df in splits.items():
        print(f"\n{name} sample:\n{df.head(2)}")


if __name__ == "__main__":
    main()
