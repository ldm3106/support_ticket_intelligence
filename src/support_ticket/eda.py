import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import REPORTS, save_json
from .data import download, prepare
from .preprocessing import text_key


def run():
    raw = download()
    frames, labels = prepare()
    summary = {}
    for name, frame in raw.items():
        words = frame.text.str.split().str.len()
        keys = frame.text.map(text_key)
        summary[name] = {"shape": list(frame.shape), "missing": frame.isna().sum().to_dict(),
                         "exact_duplicates": int(frame.duplicated().sum()),
                         "normalized_text_duplicates": int(keys.duplicated().sum()),
                         "classes": int(frame.label.nunique()), "words": words.describe().to_dict(),
                         "under_3_words": int((words < 3).sum()), "over_100_words": int((words > 100).sum())}
    summary["train_test_text_overlap"] = len(set(raw["train"].text.map(text_key)) & set(raw["test"].text.map(text_key)))
    save_json(REPORTS / "eda_summary.json", summary)
    counts = pd.DataFrame({name: frame.label.value_counts() for name, frame in frames.items()}).fillna(0).astype(int)
    counts.to_csv(REPORTS / "class_distribution.csv")
    figures = REPORTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(frames["train"].text.str.split().str.len(), bins=35, color="#246b87")
    axes[0].set(title="Training query length", xlabel="Whitespace-delimited words", ylabel="Queries")
    axes[1].bar(range(len(labels)), counts.loc[labels, "train"], color="#246b87")
    axes[1].set(title="Training examples per intent", xlabel="Intent ID (see label manifest)", ylabel="Queries")
    fig.tight_layout()
    fig.savefig(figures / "text_eda.png", dpi=160)
    plt.close(fig)
    print(summary)
    return summary


if __name__ == "__main__":
    run()
