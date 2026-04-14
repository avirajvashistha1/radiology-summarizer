"""
Phase 2: Data Cleaning and Feature Creation
Cleans the parsed reports CSV:
- Drops rows with missing findings or impression
- Normalizes whitespace
- Deduplicates exact pairs
- Logs corpus statistics

Usage:
    python src/data/clean.py --input_path data/processed/reports.csv \
                              --output_path data/cleaned/reports_clean.csv
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize whitespace; preserve casing and punctuation."""
    text = text.strip()
    # Collapse multiple spaces, tabs, and newlines into a single space
    text = re.sub(r"\s+", " ", text)
    return text


def clean_reports(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load raw parsed CSV, apply cleaning steps, and save the clean version.
    """
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} rows from '{input_path}'")

    # --- Drop rows with null or empty findings / impression ---
    df["findings"] = df["findings"].astype(str).replace("nan", None)
    df["impression"] = df["impression"].astype(str).replace("nan", None)

    df = df[df["findings"].notna() & df["impression"].notna()]

    # Strip whitespace and drop rows that are blank after stripping
    df["findings"] = df["findings"].apply(normalize_text)
    df["impression"] = df["impression"].apply(normalize_text)

    df = df[df["findings"].str.len() > 0]
    df = df[df["impression"].str.len() > 0]

    after_null_drop = len(df)
    logger.info(
        f"Dropped {initial_count - after_null_drop} rows with missing/empty fields. "
        f"Remaining: {after_null_drop}"
    )

    # --- Deduplicate exact (findings, impression) pairs ---
    df = df.drop_duplicates(subset=["findings", "impression"])
    after_dedup = len(df)
    logger.info(
        f"Dropped {after_null_drop - after_dedup} duplicate pairs. "
        f"Remaining: {after_dedup}"
    )

    df = df.reset_index(drop=True)

    # --- Corpus statistics ---
    df["findings_word_count"] = df["findings"].str.split().str.len()
    df["impression_word_count"] = df["impression"].str.split().str.len()

    logger.info("\n--- Corpus Statistics ---")
    for field in ["findings_word_count", "impression_word_count"]:
        stats = df[field].describe(percentiles=[0.5, 0.75, 0.95])
        logger.info(
            f"{field}: mean={stats['mean']:.1f}, "
            f"median={stats['50%']:.0f}, "
            f"95th_pct={stats['95%']:.0f}, "
            f"max={stats['max']:.0f}"
        )

    # Drop helper columns before saving
    df = df.drop(columns=["findings_word_count", "impression_word_count"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df.to_parquet(output_path.with_suffix(".parquet"), index=False)

    logger.info(f"Saved {len(df)} clean rows to '{output_path}'")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean parsed radiology reports")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/reports.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/cleaned/reports_clean.csv",
    )
    args = parser.parse_args()

    df = clean_reports(args.input_path, args.output_path)
    print(f"\nSample clean data:\n{df.head()}")


if __name__ == "__main__":
    main()
