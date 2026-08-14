"""Train and evaluate the CivicRoute service-request triage model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/service_requests.csv"
REVIEW_THRESHOLD = 0.58


def read_data() -> pd.DataFrame:
    return pd.read_csv(DATA).dropna(subset=["text", "service_queue"])


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


def score_model(model: Pipeline, test: pd.DataFrame) -> dict:
    predictions = model.predict(test["text"])
    probability = model.predict_proba(test["text"]).max(axis=1)
    reviewed = probability < REVIEW_THRESHOLD
    auto_correct = predictions == test["service_queue"].to_numpy()
    return {
        "accuracy": round(accuracy_score(test["service_queue"], predictions), 3),
        "macro_f1": round(f1_score(test["service_queue"], predictions, average="macro"), 3),
        "manual_review_rows": int(reviewed.sum()),
        "automatic_coverage": round(float((~reviewed).mean()), 3),
        "automatic_accuracy": round(
            float(auto_correct[~reviewed].mean()) if (~reviewed).any() else 0.0,
            3,
        ),
        "predictions": predictions,
        "probability": probability,
    }


def train_and_evaluate() -> dict:
    frame = read_data()
    train, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame["service_queue"],
    )
    model = build_model().fit(train["text"], train["service_queue"])
    random_result = score_model(model, test)
    predictions = random_result.pop("predictions")
    probability = random_result.pop("probability")
    report = classification_report(test["service_queue"], predictions, output_dict=True, zero_division=0)

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    group_train_idx, group_test_idx = next(
        group_split.split(frame, groups=frame["category_code"])
    )
    group_train = frame.iloc[group_train_idx]
    group_test = frame.iloc[group_test_idx]
    group_model = build_model().fit(group_train["text"], group_train["service_queue"])
    group_result = score_model(group_model, group_test)
    group_result.pop("predictions")
    group_result.pop("probability")
    metrics = {
        "source_rows": int(json.loads((ROOT / "data/raw/source_manifest.json").read_text())["source_rows"]),
        "model_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "service_queues": int(frame["service_queue"].nunique()),
        "accuracy": random_result["accuracy"],
        "macro_f1": random_result["macro_f1"],
        "review_threshold": REVIEW_THRESHOLD,
        "manual_review_rows": random_result["manual_review_rows"],
        "automatic_coverage": random_result["automatic_coverage"],
        "automatic_accuracy": random_result["automatic_accuracy"],
        "category_holdout_rows": int(len(group_test)),
        "category_holdout_categories": int(group_test["category_code"].nunique()),
        "category_holdout_accuracy": group_result["accuracy"],
        "category_holdout_macro_f1": group_result["macro_f1"],
        "category_holdout_automatic_coverage": group_result["automatic_coverage"],
        "category_holdout_automatic_accuracy": group_result["automatic_accuracy"],
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
    joblib.dump(model, ROOT / "models/civicroute_triage.joblib")
    (ROOT / "data/processed/evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#f4f6fa")
    ConfusionMatrixDisplay.from_predictions(
        test["service_queue"],
        predictions,
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        xticks_rotation=40,
        normalize="true",
        values_format=".2f",
    )
    axes[0].set_title("Random holdout confusion matrix")
    axes[1].hist(probability, bins=24, color="#6d4fa3", edgecolor="white")
    axes[1].axvline(REVIEW_THRESHOLD, color="#d45939", linestyle="--", label="Manual review threshold")
    axes[1].set(title="Prediction confidence", xlabel="Highest class probability", ylabel="Service requests")
    axes[1].legend()
    fig.suptitle(
        f"CivicRoute Service Request Triage | {len(frame):,} labeled requests | Category holdout F1 {metrics['category_holdout_macro_f1']:.3f}",
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
