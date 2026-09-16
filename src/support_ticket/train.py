"""Fine-tune all pretrained weights with an explicit PyTorch training loop."""
import argparse
from dataclasses import asdict, replace
from time import perf_counter
import platform

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed)

from .config import MODELS, REPORTS, TrainConfig, save_json
from .data import load_prepared
from .evaluate import evaluate_predictions


def tokenize_frame(frame, tokenizer, max_length: int):
    dataset = Dataset.from_dict({"text": frame.text.tolist(), "labels": frame.label_id.tolist()})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
                       batched=True, remove_columns=["text"])


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for batch in loader:
        truth.extend(batch.pop("labels").tolist())
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
        predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return truth, predictions


def run(config: TrainConfig):
    set_seed(config.seed)
    torch.set_num_threads(config.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, labels = load_prepared()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, revision=config.model_revision, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)})
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loaders = {name: DataLoader(tokenize_frame(frame, tokenizer, config.max_length),
                               batch_size=config.batch_size, shuffle=name == "train", collate_fn=collator)
               for name, frame in frames.items()}
    lengths = [len(ids) for ids in tokenizer(frames["train"].text.tolist(), truncation=False)["input_ids"]]
    save_json(REPORTS / "tokenization.json", {"max_length": config.max_length,
              "training_truncation_fraction": float(np.mean(np.array(lengths) > config.max_length)),
              "training_token_percentiles": np.percentile(lengths, [50, 90, 95, 99, 100]).tolist()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = len(loaders["train"]) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    best_score, history = -1.0, []
    output = MODELS / "transformer"
    start = perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch["labels"])
        truth, predictions = predict_loader(model, loaders["validation"], device)
        score = f1_score(truth, predictions, average="macro", zero_division=0)
        row = {"epoch": epoch + 1, "train_loss": total_loss / len(frames["train"]),
               "validation_macro_f1": score, "elapsed_seconds": perf_counter() - start}
        history.append(row)
        print(row, flush=True)
        if score > best_score:
            best_score = score
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
            save_json(output / "inference_config.json", {"max_length": config.max_length})
        save_json(REPORTS / "training_history.json", history)
    training_seconds = perf_counter() - start
    model = AutoModelForSequenceClassification.from_pretrained(output).to(device)
    start = perf_counter()
    truth, predictions = predict_loader(model, loaders["test"], device)
    inference_seconds = perf_counter() - start
    metrics = evaluate_predictions(truth, predictions, labels, "transformer",
                                   training_seconds=training_seconds, test_inference_seconds=inference_seconds)
    save_json(REPORTS / "training_run.json", {**asdict(config), "device": str(device),
              "platform": platform.platform(), "torch_version": torch.__version__,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_epoch": max(history, key=lambda row: row["validation_macro_f1"])["epoch"],
              "model_revision": config.model_revision})
    print(metrics, flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(replace(TrainConfig(), epochs=args.epochs))
