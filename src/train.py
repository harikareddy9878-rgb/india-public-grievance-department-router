"""Train and evaluate an interpretable grievance routing model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/grievances.csv"


def read_data() -> tuple[list[str], list[str]]:
    with DATA.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["text"] for row in rows], [row["category"] for row in rows]


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def train_and_evaluate() -> dict:
    texts, labels = read_data()
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, stratify=labels, random_state=42)
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test).max(axis=1)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "rows": len(texts),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "macro_f1": round(f1_score(y_test, predictions, average="macro"), 3),
        "mean_confidence": round(float(probabilities.mean()), 3),
        "review_threshold": 0.62,
        "low_confidence_test_cases": int((probabilities < 0.62).sum()),
        "class_metrics": {name: {"precision": round(values["precision"], 3), "recall": round(values["recall"], 3), "f1": round(values["f1-score"], 3)} for name, values in report.items() if name in model.classes_},
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "data/processed").mkdir(exist_ok=True)
    (ROOT / "evidence").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/grievance_router.joblib")
    (ROOT / "data/processed/evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#f4f6fa")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[0], cmap="Blues", colorbar=False, xticks_rotation=35)
    axes[0].set_title("Held-out confusion matrix")
    axes[1].hist(probabilities, bins=16, color="#6d4fa3", edgecolor="white")
    axes[1].axvline(metrics["review_threshold"], color="#d45939", linestyle="--", label="Manual review threshold")
    axes[1].set(title="Prediction confidence", xlabel="Highest class probability", ylabel="Test grievances")
    axes[1].legend()
    fig.suptitle(f"Consumer Grievance Router  |  Accuracy {metrics['accuracy']:.1%}  |  Macro F1 {metrics['macro_f1']:.3f}", fontsize=15, weight="bold", color="#20324c")
    fig.tight_layout()
    fig.savefig(ROOT / "evidence/routing_evaluation.png", dpi=180)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))

