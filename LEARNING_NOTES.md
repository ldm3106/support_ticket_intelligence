# Learning notes

| Concept | How it appears here |
|---|---|
| NLP | Mapping customer language to one of 77 banking intents. |
| Text preprocessing | Normalize whitespace and Unicode; preserve negation and punctuation. |
| TF-IDF | Unigram/bigram counts weighted by rarity; sparse vectors feed logistic regression. |
| Tokenization / tokens | WordPiece splits a query into vocabulary entries, including subwords and special tokens. |
| Vocabulary | The pretrained tokenizer's fixed mapping from token strings to integer IDs. |
| Embeddings | Learned dense vectors turn token IDs into neural inputs. |
| Attention | Tokens combine information from other tokens in a query. |
| Transformer | Stacked attention and feed-forward blocks produce contextual representations. |
| BERT / DistilBERT | BERT is a bidirectional encoder; DistilBERT is a distilled, smaller relative. This project uses the much smaller two-layer BERT-Tiny for CPU training. |
| Pretrained model | Language representations learned before this project; the 77-class classification head starts newly initialized. |
| Transfer learning / fine-tuning | Update all pretrained weights plus the new head on labeled BANKING77 queries. |
| Tensors / batches | Numeric arrays hold a batch of token IDs, masks and target IDs. |
| Padding / masks | Dynamic batch padding aligns lengths; attention masks identify real tokens versus padding. |
| Truncation | Cap input at 96 tokens; measured training truncation is reported. Long queries may lose relevant information. |
| Forward pass / logits | The encoder and classification head produce 77 unnormalized scores per query. |
| Loss | Cross-entropy compares logits with the correct class ID. |
| Optimizer | AdamW updates weights after backpropagation; gradient clipping limits unusually large updates. |
| Epoch | One pass through the training split. Validation follows each epoch. |
| Batch size | 32 queries per update; affects memory, noise and throughput. |
| Learning rate | Starts with warmup, then decays from the configured peak of 0.0003. |
| Softmax / confidence | Normalize logits into scores summing to one. Maximum score is not calibrated correctness probability. |
| Training vs inference | Training computes gradients with dropout active; inference disables gradients and dropout. |
| Model selection | Save the epoch with highest validation macro F1; evaluate the saved model on test afterwards. |
| Metrics | Accuracy counts correct queries; macro metrics weight each intent equally; weighted metrics weight by support. |

Trace the implementation in order: `data.py` → `preprocessing.py` → `train.py` → `evaluate.py` → `inference.py`. Hugging Face handles pretrained loading, tokenization, dataset mapping and model serialization; the short PyTorch loop makes optimization visible.
