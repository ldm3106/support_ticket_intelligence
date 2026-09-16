# Project decisions

| Decision / problem | Options | Choice and why | Trade-offs | Validation |
|---|---|---|---|---|
| Credible support labels | Unclear-license helpdesk dumps; public customer-service benchmarks | BANKING77 with published source and CC BY 4.0 attribution | Banking utterances rather than full tickets; 77 fine-grained intents | Source CSV checksums, EDA and label manifest |
| Lightweight comparison | Multiple classical models; one baseline | TF-IDF unigram/bigram + logistic regression, C=4 | Lexical features lack contextual understanding but can be strong on small data | Same training/test split and metrics |
| Local Transformer compute | DistilBERT; BERT-Tiny | Google's pretrained two-layer, 128-hidden BERT; approximately 16 GB RAM and no detected NVIDIA tooling | Less capacity; does not measure all Transformers | Actual CPU training and validation history |
| Text handling | Stemming/stopwords; minimal normalization | Preserve language for pretrained tokenizer | More surface variation; negation retained | Preprocessing tests |
| Length cap | 64, 96, 128 | 96 tokens as a modest initial CPU budget | Some long text may be cut | Report training token percentiles and truncation fraction |
| Split leakage | Random whole-dataset split; official test | Keep official test; deduplicate development data and remove overlaps before stratified 15% validation | Does not remove semantic duplicates | Disjointness tests and manifest |
| Evaluation | Accuracy alone; macro and class reports | Accuracy, macro/weighted precision, recall, F1; full confusion matrices | Single split does not quantify sampling uncertainty | Saved test predictions and class-level reports |
| Fine-tuning | Frozen encoder; end-to-end updates | All weights, 12 epochs, batch 32, AdamW 3e-4, weight decay .01, 10% warmup, linear decay, clipping 1.0 | CPU cost; one configuration, not an exhaustive search | Best validation macro F1 checkpoint reloaded for test |
| Training API | Trainer; explicit PyTorch loop | Compact loop plus Hugging Face model/tokenizer/Datasets APIs | Resume state not implemented | Integration test and actual run |
| Packaging | Loose scripts; installable src package | `src/support_ticket`, editable installation | Small setup step | Module CLI execution |

Only the best model/tokenizer is retained. Optimizer checkpoints and interrupted-run resume are postponed. Random seed 42 fixes the split and random initialization; hardware/library differences may still affect numerical reproducibility. Raw data hashes are fixed; dependency versions are recorded in the lock file. Public pretrained model revision is recorded in run metadata when available.

The initial download resolved to `30b0a37ccaaa32f332884b96992754e246e48c5f`, verified from the local Hugging Face snapshot path. Configuration now pins that exact revision for both tokenizer and model. Training supports range from 30 to 159 examples per class; retain unweighted loss for this initial comparison and examine macro metrics before considering weighting.
