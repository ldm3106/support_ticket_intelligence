from time import perf_counter
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .config import MODELS, REPORTS, save_json
from .data import load_prepared
from .evaluate import evaluate_predictions


def run():
    frames, labels = load_prepared()
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=4.0, max_iter=1000, random_state=42))
    start = perf_counter()
    model.fit(frames["train"].text, frames["train"].label_id)
    training_seconds = perf_counter() - start
    start = perf_counter()
    predictions = model.predict(frames["test"].text)
    inference_seconds = perf_counter() - start
    metrics = evaluate_predictions(frames["test"].label_id, predictions, labels, "baseline",
                                   training_seconds=training_seconds, test_inference_seconds=inference_seconds)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "baseline.joblib")
    save_json(MODELS / "baseline_labels.json", labels)
    print(metrics)
    return metrics


if __name__ == "__main__":
    run()
