# BANKING77 data

Source: [PolyAI BANKING77 dataset card](https://huggingface.co/datasets/PolyAI/banking77).
Original files: [PolyAI task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
Attribution: Casanueva, Temcinas, Gerz, Henderson and Vulic (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP for ConvAI, ACL.
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The release contains 10,003 training and 3,080 test customer-service queries in English, with 77 banking intents. These are customer-service utterances, not complete helpdesk conversations; no priority target exists. Source columns are `text` and `category`. The code renames category to `label` and adds normalized `key` and integer `label_id`.

`python -m support_ticket.data` downloads the original CSVs without credentials and verifies SHA-256 against the published Hugging Face metadata. Raw and processed files are excluded from Git. Hugging Face Datasets is used for Transformer tokenization; direct source CSV ingestion avoids dependence on legacy dataset loading scripts.

The official test set is retained. Before stratifying the remaining training data 85/15, normalized duplicate training queries, ambiguous duplicate training keys, and training queries also present in test are removed. Normalization is NFKC, whitespace collapse and case folding for duplicate detection. Model input retains case and punctuation; the pretrained tokenizer performs its own uncasing. Near-duplicates and semantic overlaps are not removed.

See `reports/split_manifest.json` for all 77 exact labels, checksums, counts and seed. `reports/class_distribution.csv` lists every class count by split; `reports/eda_summary.json` gives missing values, duplicates and length statistics. These outputs are produced from downloaded data, not manually entered. Label examples include `card_arrival`, `transaction_charged_twice` and `passcode_forgotten`.

Limitations: English banking only; single intent per short query; no real queue assignments, urgency, multilingual coverage, temporal split or out-of-domain labels. Source documentation does not establish that every record is an unmodified production ticket. Treat this as a published customer-service benchmark, not evidence of production traffic performance. Review licensing and privacy again before combining it with proprietary customer text.
