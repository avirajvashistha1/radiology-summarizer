"""
Phase 1: Data Processing
Parses raw XML radiology report files from the Open-i NLM dataset.
Extracts FINDINGS and IMPRESSION fields into a flat CSV.

Usage:
    python src/data/parse_xml.py --input_dir data/raw --output_path data/processed/reports.csv
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_fields(xml_path: Path) -> dict:
    """
    Parse a single XML report file and extract FINDINGS and IMPRESSION text.
    Returns a dict with keys: report_id, findings, impression.
    Returns None if the file cannot be parsed.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {xml_path.name}: {e}")
        return None

    root = tree.getroot()

    findings_parts = []
    impression_parts = []

    # AbstractText nodes live under MedlineCitation/Article/Abstract/AbstractText
    # but the exact nesting can vary — search the whole tree
    for node in root.iter("AbstractText"):
        label = node.get("Label", "").upper()
        text = (node.text or "").strip()
        if not text:
            continue
        if label == "FINDINGS":
            findings_parts.append(text)
        elif label == "IMPRESSION":
            impression_parts.append(text)

    return {
        "report_id": xml_path.stem,
        "findings": " ".join(findings_parts) if findings_parts else None,
        "impression": " ".join(impression_parts) if impression_parts else None,
    }


def parse_corpus(input_dir: str, output_path: str) -> pd.DataFrame:
    """
    Iterate over all XML files in input_dir, extract fields, and save to CSV.
    """
    xml_files = sorted(Path(input_dir).glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in '{input_dir}'. "
            "Please extract NLMCXR_reports.tgz into the data/raw/ directory."
        )

    logger.info(f"Found {len(xml_files)} XML files in '{input_dir}'")

    records = []
    failed = 0
    for xml_path in xml_files:
        record = extract_fields(xml_path)
        if record is None:
            failed += 1
            continue
        records.append(record)

    logger.info(f"Parsed {len(records)} files successfully, {failed} failed")

    df = pd.DataFrame(records, columns=["report_id", "findings", "impression"])

    # Count rows with missing fields (keep them for EDA visibility)
    missing_findings = df["findings"].isna().sum()
    missing_impression = df["impression"].isna().sum()
    logger.info(
        f"Rows with no FINDINGS: {missing_findings}, no IMPRESSION: {missing_impression}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Also save Parquet for fast downstream reading
    parquet_path = output_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Saved {len(df)} rows to '{output_path}' and '{parquet_path}'")
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse Open-i XML radiology reports")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing XML report files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/reports.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    df = parse_corpus(args.input_dir, args.output_path)
    print(f"\nSample output:\n{df.head()}")
    print(f"\nShape: {df.shape}")


if __name__ == "__main__":
    main()
