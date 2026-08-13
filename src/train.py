"""Train and evaluate an interpretable CPGRAMS department router."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/grievances.csv"
REVIEW_THRESHOLD = 0.58


def read_data() -> pd.DataFrame:
    return pd.read_csv(DATA).dropna(subset=["text", "department"])


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.98,
                    max_features=70_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=700,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate() -> dict:
    frame = read_data()
    train, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame["department"],
    )
    model = build_model()
    model.fit(train["text"], train["department"])
    predictions = model.predict(test["text"])
    probability = model.predict_proba(test["text"]).max(axis=1)
    report = classification_report(test["department"], predictions, output_dict=True, zero_division=0)
    reviewed = probability < REVIEW_THRESHOLD
    auto_correct = predictions == test["department"].to_numpy()
    coverage = float((~reviewed).mean())
    auto_accuracy = float(auto_correct[~reviewed].mean()) if (~reviewed).any() else 0.0
    metrics = {
        "source_rows": int(json.loads((ROOT / "data/raw/source_manifest.json").read_text())["source_rows"]),
        "model_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "departments": int(frame["department"].nunique()),
        "accuracy": round(accuracy_score(test["department"], predictions), 3),
        "macro_f1": round(f1_score(test["department"], predictions, average="macro"), 3),
        "review_threshold": REVIEW_THRESHOLD,
        "manual_review_rows": int(reviewed.sum()),
        "automatic_coverage": round(coverage, 3),
        "automatic_accuracy": round(auto_accuracy, 3),
        "class_metrics": {
            name: {
                "precision": round(values["precision"], 3),
                "recall": round(values["recall"], 3),
                "f1": round(values["f1-score"], 3),
                "support": int(values["support"]),
            }
            for name, values in report.items()
            if name in model.classes_
        },
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "evidence").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/grievance_router.joblib")
    (ROOT / "data/processed/evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#f4f6fa")
    ConfusionMatrixDisplay.from_predictions(
        test["department"],
        predictions,
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        xticks_rotation=40,
        normalize="true",
        values_format=".2f",
    )
    axes[0].set_title("Normalised held-out confusion matrix")
    axes[1].hist(probability, bins=24, color="#6d4fa3", edgecolor="white")
    axes[1].axvline(REVIEW_THRESHOLD, color="#d45939", linestyle="--", label="Manual review threshold")
    axes[1].set(title="Prediction confidence", xlabel="Highest class probability", ylabel="Test grievances")
    axes[1].legend()
    fig.suptitle(
        f"CPGRAMS Department Router | {len(frame):,} labeled grievances | Macro F1 {metrics['macro_f1']:.3f}",
        fontsize=16,
        weight="bold",
        color="#20324c",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "evidence/routing_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
