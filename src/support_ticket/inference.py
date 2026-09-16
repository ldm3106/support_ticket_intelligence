import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import MODELS
from .preprocessing import prepare_text


class TicketPredictor:
    """Load once and reuse for repeated local predictions."""

    def __init__(self, model_dir: Path = MODELS / "transformer"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.max_length = json.loads((Path(model_dir) / "inference_config.json").read_text())["max_length"]

    @torch.inference_mode()
    def predict_ticket(self, text: str) -> dict:
        text = prepare_text(text)
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        probabilities = self.model(**inputs).logits.softmax(dim=-1)[0]
        confidence, index = probabilities.max(dim=-1)
        return {"predicted_category": self.model.config.id2label[index.item()],
                "confidence_score": float(confidence),
                "truncated": len(self.tokenizer.encode(text)) > self.max_length}


def predict_ticket(text: str) -> dict:
    return TicketPredictor().predict_ticket(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(predict_ticket(args.text), indent=2))
