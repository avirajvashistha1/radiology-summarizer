#!/usr/bin/env bash
# run_pipeline.sh — End-to-end reproducibility script
# Runs all phases from raw data to evaluation results.
# Prerequisites: raw XML files extracted into data/raw/
#
# Usage:
#   bash scripts/run_pipeline.sh

set -e  # Exit immediately on any error

echo "=============================================="
echo "  Radiology Summarizer — Full Pipeline"
echo "=============================================="

# Phase 1: Parse XML
echo ""
echo "[1/6] Parsing XML reports..."
python src/data/parse_xml.py \
  --input_dir data/raw \
  --output_path data/processed/reports.csv

# Phase 2: Clean data
echo ""
echo "[2/6] Cleaning data..."
python src/data/clean.py \
  --input_path data/processed/reports.csv \
  --output_path data/cleaned/reports_clean.csv

# Phase 5: Split data
echo ""
echo "[3/6] Splitting into train/val/test..."
python src/data/split.py \
  --input_path data/cleaned/reports_clean.csv \
  --output_dir data/cleaned

# Phase 4: Tokenize
echo ""
echo "[4/6] Tokenizing datasets..."
python src/modeling/tokenize_dataset.py \
  --data_dir data/cleaned \
  --output_dir data/tokenized \
  --model_name facebook/bart-base

# Phase 6: Train
echo ""
echo "[5/6] Fine-tuning model (this may take 60-90 minutes on GPU)..."
python src/modeling/train.py \
  --tokenized_dir data/tokenized \
  --output_dir model \
  --model_name facebook/bart-base

# Phase 7: Evaluate
echo ""
echo "[6/6] Evaluating model on test set..."
python src/modeling/evaluate.py \
  --model_dir model \
  --tokenized_dir data/tokenized \
  --results_dir results

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Metrics saved to: results/eval_metrics.json"
echo "  Sample predictions: results/sample_predictions.json"
echo "=============================================="
