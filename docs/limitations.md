# Limitations and Recommendations

## Implementation Limitations

**Consumer PC constraint — reduced model capacity:**
The model was fine-tuned on a standard consumer PC using `facebook/bart-base` (139M parameters)
with fp16 mixed-precision training and gradient accumulation to fit within 8 GB VRAM. A full
production implementation with multiple GPUs would allow fine-tuning of larger, more capable
models such as `facebook/bart-large` (406M parameters) or clinical domain-specific models like
`microsoft/BiomedNLP-PubMedBERT-base` or `allenai/led-base-16384`, which would likely yield
ROUGE scores of 0.45–0.55 and produce more clinically accurate impressions.

**Dataset size:**
The Open-i corpus contains approximately 3,400–3,600 usable pairs — a relatively small
dataset for fine-tuning a summarization model. Augmenting with additional radiology report
datasets (e.g., MIMIC-CXR, which contains over 200,000 reports) would significantly improve
generalization and output quality.

**Single-institution data:**
All training data originates from a single institution's historical reports. The model's output
style and vocabulary may not match conventions at other institutions.

**ROUGE metric limitations:**
ROUGE measures lexical overlap between generated and reference text — it does not assess
clinical accuracy, factual correctness, or patient safety. A high ROUGE score does not guarantee
that generated impressions are medically accurate. **This model must NOT be used for clinical
decision-making without expert radiologist review.**

**Input length:**
Findings text longer than 512 tokens will be silently truncated. This affects a small minority
of reports (< 1%) but could cause important findings to be omitted from the generated impression.

**Out-of-distribution inputs:**
The model was trained exclusively on chest X-ray reports. It will produce unreliable outputs for:
- Other imaging modalities (CT, MRI, ultrasound)
- Non-English text
- Non-radiology medical text

## Recommendations for Improvement

1. **Scale up model size** with multi-GPU training: use `facebook/bart-large` or `google/pegasus-large`
2. **Expand training data** with MIMIC-CXR or RadGraph-annotated datasets
3. **Add clinical validation**: have radiologists rate a sample of model outputs for accuracy
4. **Implement input length handling**: split long findings into chunks and aggregate summaries
5. **Add confidence scoring**: flag low-confidence outputs for mandatory human review
6. **Continuous fine-tuning**: periodically re-train on new reports to maintain relevance
