# Interview notes

- Built a reproducible customer-service intent classifier: validated public data → lexical baseline → pretrained Transformer fine-tuning → held-out evaluation → saved-model inference.
- Business use: suggest a banking support intent to a routing service. Actual routing, priority prediction and ticket-platform integration are not implemented.
- Hugging Face provides consistent tokenizer/model APIs and portable saved artifacts. Pretraining reduces the need to learn language from this small labeled dataset.
- BERT-Tiny makes all-weight fine-tuning practical on CPU. It has less representation capacity than DistilBERT; results here do not establish which architecture is generally best.
- The baseline tests whether Transformer complexity produces measurable benefit. Report the actual comparison even if the baseline wins.
- Explain split hygiene, duplicate removal, validation-only checkpoint selection, macro F1 and confused class pairs using reports.
- Training time excludes download/tokenization; batched inference timing excludes tokenization for the Transformer. These are local workload measurements, not serving latency SLAs.
- Confidence is an uncalibrated softmax maximum. A future review threshold requires validation on representative traffic and costs of misrouting.

Questions to revisit later:

1. Why preserve stopwords and negation for BERT?
2. What does the attention mask do, and why use dynamic padding?
3. What is newly initialized versus pretrained?
4. Why can TF-IDF beat a small Transformer?
5. How would you detect unknown intents, drift and PII?
6. What changes for multi-intent tickets or long conversations?
7. How would you measure per-request latency and calibrate confidence?
8. Which observed class confusions would you investigate first?
