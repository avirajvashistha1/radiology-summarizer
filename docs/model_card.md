# Model Card — Radiology Report Summarizer

## Model Description
Fine-tuned `facebook/bart-base` (139M parameters) for abstractive summarization of
chest X-ray radiology reports. Given free-text FINDINGS, the model generates a concise
IMPRESSION summary.

## Intended Use
Assisting radiologists by generating draft IMPRESSION sections from dictated or typed
FINDINGS text. **Not for clinical decision-making without radiologist review.**

## Training Data
Open-i NLM Chest X-ray dataset (~3,400–3,600 usable FINDINGS/IMPRESSION pairs after cleaning).

## Training Procedure
| Hyperparameter | Value |
|---|---|
| Base model | facebook/bart-base |
| Epochs | 5 |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Learning rate | 3e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max source length | 512 tokens |
| Max target length | 128 tokens |
| Decoding | Beam search (num_beams=4) |
| Train/Val/Test split | 72% / 8% / 20% |

## Evaluation Results
*(Fill in after training)*

| Metric | Score |
|---|---|
| ROUGE-1 F1 | TBD |
| ROUGE-2 F1 | TBD |
| ROUGE-L F1 | TBD |

## Sample Predictions
*(Fill in after evaluation — see results/sample_predictions.json)*

## Limitations
See `docs/limitations.md`
