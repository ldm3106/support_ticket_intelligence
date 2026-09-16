# Project progress

## Completed

- Preserved existing insurance and credit-risk projects; implemented a separate `support-ticket-intelligence` project.
- Downloaded checksum-verified BANKING77 public source data with attribution/license documentation.
- Executed EDA and notebook: no missing values; 11 development rows removed through normalization/deduplication and test-overlap filtering.
- Prepared 8,493 training / 1,499 validation / 3,080 test queries across 77 labels.
- Trained and saved TF-IDF + logistic regression: test accuracy 0.883117, macro F1 0.883256.
- Loaded real pretrained Google BERT-Tiny/tokenizer and fine-tuned all weights for 12 epochs on CPU.
- Selected epoch 12 by validation macro F1 (0.857754), saved and reloaded model/tokenizer, then evaluated test once: accuracy 0.872727, macro F1 0.869269.
- Recorded actual training time: baseline 6.76 seconds, Transformer 352.14 seconds; inference timing scope documented.
- Saved full classification reports, true/predicted IDs, per-class metrics, confusion matrices and actual inference examples.
- Executed notebook; inspected EDA and Transformer confusion-matrix figures.
- Final test suite: **8 passed**, including real saved-model tokenization and inference; no skips in the completed run.
- Dependency consistency (`pip check`) passed; Python source compilation passed.
- README, learning notes, decisions, interview notes and measured comparison written.
- Exact dependency lock recorded; pretrained checkpoint revision pinned from the actual cached snapshot.

## Current

Local implementation and measured experiment complete. Files are ready for a first commit. No remote publishing or deployment was performed.

## Issues and limitations

- Baseline outperformed this BERT-Tiny configuration; preserve both results without claiming Transformer superiority.
- `virtual_card_not_working` has particularly poor Transformer recall (0.075); see class confusion analysis before any routing use.
- Training intents have unequal support; unweighted loss was used. No extensive tuning or repeated-seed uncertainty analysis.
- English banking queries only; unrelated input still receives a known banking label. Confidence is uncalibrated.
- Model artifacts and data are intentionally ignored by Git; a fresh clone needs training before local inference.
- CPU runtime is a local measurement, not production latency. No API, deployment, PII filter, calibrated abstention or routing integration.

## Next / user action

Create an empty GitHub repository named `support-ticket-intelligence`, then commit and push this directory using the README commands. No credentials are needed for local use. Run commands from this project directory to avoid committing it into the parent insurance repository.

## Postponed improvements

DistilBERT comparison on suitable compute; validation-driven imbalance treatment; multiple-seed evaluation; calibration and unknown-intent rejection; human review; monitoring; privacy controls; API/container/ticket-system integration. None is claimed as implemented.
