"""Generate a reviewable comparison directly from measured artifacts."""
import json
from pathlib import Path
import pandas as pd
from support_ticket.config import REPORTS


def run():
    metrics = {name: json.loads((REPORTS / f"{name}_metrics.json").read_text())
               for name in ("baseline", "transformer")}
    rows = ["# Measured model comparison", "", "Both models use the same 8,493 training queries and official 3,080-query test set. Transformer checkpoint selection uses 1,499 validation queries.", "",
            "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Train seconds | Test batch seconds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in metrics.items():
        numbers = [values[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1-score", "weighted_f1-score", "training_seconds", "test_inference_seconds")]
        rows.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in numbers) + " |")
    delta = metrics["transformer"]["macro_f1-score"] - metrics["baseline"]["macro_f1-score"]
    rows.extend(["", f"Transformer minus baseline macro F1: {delta:+.4f}. This comparison describes this small model and configuration, not all Transformers.", "",
                 "Training timing excludes download and tokenization; Transformer time includes validation/checkpoint saves. Test timing is one local batch workload: baseline includes TF-IDF transformation, Transformer excludes prior tokenization. These timings are not an equal end-to-end serving benchmark, and do not include model loading. Accuracy equals macro recall because test supports are balanced (40 each); macro F1 still captures precision/recall differences."])
    for name in metrics:
        report = json.loads((REPORTS / f"{name}_classification_report.json").read_text())
        classes = {key: value for key, value in report.items() if key not in ("accuracy", "macro avg", "weighted avg")}
        worst = sorted(classes, key=lambda key: classes[key]["f1-score"])[:5]
        rows.extend(["", f"## {name}: lowest test F1", ""])
        rows.extend(f"- `{key}`: {classes[key]['f1-score']:.3f} (precision {classes[key]['precision']:.3f}, recall {classes[key]['recall']:.3f})." for key in worst)
        matrix = pd.read_csv(REPORTS / f"{name}_confusion_matrix.csv", index_col=0)
        pairs = sorted(((int(matrix.iloc[i, j]), matrix.index[i], matrix.columns[j])
                        for i in range(len(matrix)) for j in range(len(matrix)) if i != j), reverse=True)[:5]
        rows.extend(["", "Most frequent directional confusions:", ""])
        rows.extend(f"- `{truth}` → `{predicted}`: {count} queries." for count, truth, predicted in pairs)
    rows.extend(["", "## Practical trade-offs", "",
                 "TF-IDF logistic regression is inexpensive, with inspectable token coefficients. BERT learns contextual representations and can adapt pretrained language features, but requires tokenization, neural optimization and larger model artifacts. Attention weights alone are not a reliable explanation. Neither system is calibrated or equipped with an unknown-intent class. Class-level errors should guide any later data review; this run did not tune against test errors.", ""])
    (REPORTS / "MODEL_COMPARISON.md").write_text("\n".join(rows), encoding="utf-8")
    root = REPORTS.parent
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    table = "| Model | Test accuracy | Macro F1 | Training seconds |\n|---|---:|---:|---:|\n"
    for name, values in metrics.items():
        table += f"| {name} | {values['accuracy']:.2%} | {values['macro_f1-score']:.4f} | {values['training_seconds']:.2f} |\n"
    start, end = "<!-- RESULTS -->", "<!-- END RESULTS -->"
    before, rest = readme.split(start)
    _, after = rest.split(end)
    readme = before + start + "\n" + table + "\n" + end + after
    examples = json.loads((REPORTS / "example_predictions.json").read_text())
    example = examples[0]
    start, end = "<!-- EXAMPLE -->", "<!-- END EXAMPLE -->"
    before, rest = readme.split(start)
    _, after = rest.split(end)
    readme = before + start + "\n```json\n" + json.dumps(example, indent=2) + "\n```\n" + end + after
    readme_path.write_text(readme, encoding="utf-8")
    print("\n".join(rows[:10]))


if __name__ == "__main__":
    run()
