import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .config import REPORTS, save_json


def evaluate_predictions(y_true, y_pred, labels: list[str], name: str, **cost):
    report = classification_report(y_true, y_pred, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    metrics = {"accuracy": accuracy_score(y_true, y_pred),
               **{f"{average}_{metric}": report[f"{average} avg"][metric]
                  for average in ("macro", "weighted") for metric in ("precision", "recall", "f1-score")},
               **cost}
    save_json(REPORTS / f"{name}_metrics.json", metrics)
    save_json(REPORTS / f"{name}_classification_report.json", report)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(REPORTS / f"{name}_confusion_matrix.csv")
    figures = REPORTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    plot = ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Predicted intent ID", ylabel="True intent ID", title=f"{name}: test confusion matrix")
    fig.colorbar(plot, ax=ax, label="Queries")
    fig.tight_layout()
    fig.savefig(figures / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    # Detailed names remain readable in the CSV and class report.
    pd.DataFrame({"true_id": np.asarray(y_true), "predicted_id": np.asarray(y_pred)}).to_csv(
        REPORTS / f"{name}_predictions.csv", index=False)
    return metrics
