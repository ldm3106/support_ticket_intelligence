from dataclasses import dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    model_revision: str = "30b0a37ccaaa32f332884b96992754e246e48c5f"
    seed: int = 42
    max_length: int = 96
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    threads: int = 4


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
