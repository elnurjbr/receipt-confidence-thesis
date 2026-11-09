# Confidence-based receipt data extraction and validation

Bachelor thesis prototype, Elnur Jabbarov, Riga Technical University (supervisor Prof. I. Eriņš).

The system reads a receipt twice with Gemini (once from the Tesseract OCR text and once from the image),
checks the result with business rules R1–R4, calculates a confidence score for each field and decides:
approve automatically, re-read the uncertain fields once, or send the receipt to manual review.
Four strategies are compared on the same extraction: S0 (no checks), S1 (rules only),
S2 (confidence routing) and S3 (confidence routing with one selective re-read and guardrails).

## Files
- `receipt_pipeline.py` – extraction (OCR + Gemini, cached), validator, confidence routing S0–S3, evaluator
- `test_pipeline.py` – logic tests without API (wrong total, missing fields, duplicate items, examples A and B, guardrails G1 and G2)
- `week1_prototype.ipynb` – Google Colab notebook for the development run
- `results/run1_summary.csv`, `results/run2_summary.csv` – S0–S3 results of development runs 1 and 2

## How to run
1. Open `week1_prototype.ipynb` in Google Colab (File → Upload notebook).
2. Runtime → Run all. Paste your own Gemini API key when asked (free key from aistudio.google.com).
3. At the end, `run2_outputs.zip` is downloaded (raw Gemini answers, results, summary table).
4. Logic tests without API: `python test_pipeline.py`

## Settings used
- Model: `gemini-3.5-flash-lite`, free tier (15 requests/min, 500 requests/day)
- Paid-tier prices used for the cost estimate: $0.30 input / $2.50 output per 1M tokens (checked 2026-11-08)
- Development sample: 40 receipts from the CORD v2 **training** split (seed 42). The test split is not used.

## Notes
- Ground truth is used only by the evaluator, never by the validator or the routing.
- All weights, error probabilities and costs are provisional assumptions, not calibrated values.
- No API key is stored in this repository.
