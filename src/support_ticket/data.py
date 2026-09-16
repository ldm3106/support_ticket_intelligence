"""Download public source CSVs; protect the official test set from leakage."""
import hashlib
import json
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA, REPORTS, save_json
from .preprocessing import prepare_text, text_key

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
CHECKSUMS = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def download() -> dict[str, pd.DataFrame]:
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split, checksum in CHECKSUMS.items():
        path = raw / f"{split}.csv"
        if not path.exists():
            with urlopen(f"{SOURCE}/{split}.csv", timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"Source checksum changed: {split}; review before training.")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Cached dataset checksum mismatch: {path}")
        frames[split] = pd.read_csv(path).rename(columns={"category": "label"})
    return frames


def split_data(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    train, test = train.copy(), test.copy()
    for frame in (train, test):
        if frame[["text", "label"]].isna().any().any():
            raise ValueError("Missing text or labels require review.")
        frame["text"] = frame.text.map(prepare_text)
        frame["key"] = frame.text.map(text_key)
    # Remove ambiguous training duplicates and every training overlap with test.
    conflicts = train.groupby("key").label.nunique()
    train = train[~train.key.isin(conflicts[conflicts > 1].index)]
    train = train.drop_duplicates("key")
    train = train[~train.key.isin(test.key)]
    labels = sorted(train.label.unique().tolist())
    if not set(test.label).issubset(labels):
        raise ValueError("Test set contains unknown labels.")
    fit, validation = train_test_split(train, test_size=0.15, stratify=train.label, random_state=seed)
    mapping = {name: i for i, name in enumerate(labels)}
    frames = {}
    for name, frame in (("train", fit), ("validation", validation), ("test", test)):
        frame = frame.copy()
        frame["label_id"] = frame.label.map(mapping)
        frames[name] = frame.reset_index(drop=True)
    return frames, labels


def prepare(seed: int = 42):
    raw = download()
    frames, labels = split_data(raw["train"], raw["test"], seed)
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_json(processed / f"{name}.jsonl", orient="records", lines=True)
    save_json(processed / "labels.json", labels)
    save_json(REPORTS / "split_manifest.json", {
        "seed": seed, "source": SOURCE, "sha256": CHECKSUMS,
        "raw_counts": {k: len(v) for k, v in raw.items()},
        "split_counts": {k: len(v) for k, v in frames.items()},
        "removed_training_rows": len(raw["train"]) - len(frames["train"]) - len(frames["validation"]),
        "labels": labels,
    })
    return frames, labels


def load_prepared():
    processed = DATA / "processed"
    if not (processed / "labels.json").exists():
        return prepare()
    frames = {name: pd.read_json(processed / f"{name}.jsonl", lines=True)
              for name in ("train", "validation", "test")}
    return frames, json.loads((processed / "labels.json").read_text())


if __name__ == "__main__":
    frames, labels = prepare()
    print({name: len(frame) for name, frame in frames.items()}, "classes:", len(labels))
