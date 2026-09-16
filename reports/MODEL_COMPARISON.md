# Measured model comparison

Both models use the same 8,493 training queries and official 3,080-query test set. Transformer checkpoint selection uses 1,499 validation queries.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Train seconds | Test batch seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.8831 | 0.8893 | 0.8831 | 0.8833 | 0.8833 | 6.7585 | 0.0842 |
| transformer | 0.8727 | 0.8811 | 0.8727 | 0.8693 | 0.8693 | 352.1365 | 1.4807 |

Transformer minus baseline macro F1: -0.0140. This comparison describes this small model and configuration, not all Transformers.

Training timing excludes download and tokenization; Transformer time includes validation/checkpoint saves. Test timing is one local batch workload: baseline includes TF-IDF transformation, Transformer excludes prior tokenization. These timings are not an equal end-to-end serving benchmark, and do not include model loading. Accuracy equals macro recall because test supports are balanced (40 each); macro F1 still captures precision/recall differences.

## baseline: lowest test F1

- `pending_transfer`: 0.694 (precision 0.781, recall 0.625).
- `balance_not_updated_after_bank_transfer`: 0.719 (precision 0.653, recall 0.800).
- `card_not_working`: 0.731 (precision 0.642, recall 0.850).
- `topping_up_by_card`: 0.740 (precision 0.818, recall 0.675).
- `card_payment_not_recognised`: 0.767 (precision 0.848, recall 0.700).

Most frequent directional confusions:

- `virtual_card_not_working` → `getting_virtual_card`: 6 queries.
- `verify_my_identity` → `why_verify_identity`: 5 queries.
- `unable_to_verify_identity` → `verify_my_identity`: 5 queries.
- `top_up_reverted` → `top_up_failed`: 5 queries.
- `pin_blocked` → `get_physical_card`: 5 queries.

## transformer: lowest test F1

- `virtual_card_not_working`: 0.140 (precision 1.000, recall 0.075).
- `why_verify_identity`: 0.554 (precision 0.535, recall 0.575).
- `verify_my_identity`: 0.667 (precision 0.659, recall 0.675).
- `get_disposable_virtual_card`: 0.681 (precision 0.608, recall 0.775).
- `balance_not_updated_after_bank_transfer`: 0.744 (precision 0.696, recall 0.800).

Most frequent directional confusions:

- `virtual_card_not_working` → `get_disposable_virtual_card`: 16 queries.
- `virtual_card_not_working` → `getting_virtual_card`: 15 queries.
- `why_verify_identity` → `verify_my_identity`: 14 queries.
- `verify_my_identity` → `why_verify_identity`: 9 queries.
- `unable_to_verify_identity` → `why_verify_identity`: 9 queries.

## Practical trade-offs

TF-IDF logistic regression is inexpensive, with inspectable token coefficients. BERT learns contextual representations and can adapt pretrained language features, but requires tokenization, neural optimization and larger model artifacts. Attention weights alone are not a reliable explanation. Neither system is calibrated or equipped with an unknown-intent class. Class-level errors should guide any later data review; this run did not tune against test errors.
