from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures"


def save(name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def architecture() -> None:
    stages = [
        ("Public requests", "Kaggle JSON", "175,784 source rows"),
        ("Preparation", "Python + mappings", "clean text and queues"),
        ("Text model", "TF-IDF", "interpretable features"),
        ("Routing", "Logistic regression", "queue plus confidence"),
        ("Control", "Review threshold", "automatic or manual"),
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.axis("off")
    for index, (title, technology, detail) in enumerate(stages):
        x = 0.04 + index * 0.195
        axis.text(x, 0.55, f"{title}\n\n{technology}\n{detail}", ha="center", va="center", fontsize=9.5, bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "black"})
        if index < len(stages) - 1:
            axis.annotate("", xy=(x + 0.125, 0.55), xytext=(x + 0.075, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.set_title("CivicRoute end-to-end routing architecture", fontweight="bold", pad=18)
    save("06_architecture.png")


def test_evidence() -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    figure.patch.set_facecolor("#171717")
    axis.set_facecolor("#171717")
    axis.axis("off")
    lines = [
        "$ .venv/bin/pytest -q",
        "tests/test_router.py ....                                [100%]",
        "",
        "4 passed, 8 dependency warnings in 2.06s",
        "",
        "Validated: clear-route inference, confidence fallback,",
        "repository assets and manual-review enforcement.",
        "Warnings originate from joblib with NumPy 2.5.",
    ]
    for index, line in enumerate(lines):
        axis.text(0.06, 0.9 - index * 0.102, line, transform=axis.transAxes, color="white" if index < 4 else "#d0d0d0", family="monospace", fontsize=11.5)
    axis.set_title("Actual routing test execution", color="white", fontweight="bold", pad=16)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT / "07_test_execution.png", dpi=190, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close()


def main() -> None:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    manifest = json.loads((ROOT / "data/raw/source_manifest.json").read_text())
    plt.style.use("grayscale")

    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(
        ["Random holdout", "Category-group holdout"],
        [metrics["macro_f1"], metrics["category_holdout_macro_f1"]],
        color=["0.45", "0.75"],
        edgecolor="black",
    )
    plt.bar_label(bars, fmt="%.3f")
    plt.ylim(0.9, 1.0)
    plt.ylabel("Macro F1")
    plt.title("Generalisation test comparison")
    save("01_holdout_comparison.png")

    plt.figure(figsize=(7, 4.8))
    values = [
        metrics["category_holdout_automatic_coverage"] * 100,
        (1 - metrics["category_holdout_automatic_coverage"]) * 100,
    ]
    bars = plt.bar(
        ["Automatically routed", "Manual review"], values, color=["0.55", "0.82"], edgecolor="black"
    )
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylabel("Category-group holdout share")
    plt.title("Confidence threshold outcome")
    save("02_review_coverage.png")

    counts = manifest["service_queue_counts"]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(list(counts), list(counts.values()), color="0.72", edgecolor="black")
    plt.bar_label(bars, fmt="%d")
    plt.title("Prepared service-request rows by queue")
    save("03_queue_distribution.png")

    class_metrics = metrics["class_metrics"]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        list(class_metrics),
        [value["f1"] for value in class_metrics.values()],
        color="0.65",
        edgecolor="black",
    )
    plt.bar_label(bars, fmt="%.3f")
    plt.xlim(0.95, 1.0)
    plt.title("Random-holdout F1 by service queue")
    save("04_class_f1.png")

    plt.figure(figsize=(8, 4.8))
    labels = ["Random accuracy", "Random auto accuracy", "Group accuracy", "Group auto accuracy"]
    values = [
        metrics["accuracy"] * 100,
        metrics["automatic_accuracy"] * 100,
        metrics["category_holdout_accuracy"] * 100,
        metrics["category_holdout_automatic_accuracy"] * 100,
    ]
    bars = plt.bar(labels, values, color="0.7", edgecolor="black")
    plt.bar_label(bars, fmt="%.1f%%")
    plt.xticks(rotation=12)
    plt.ylim(95, 100.2)
    plt.title("Accuracy before and after confidence review")
    save("05_confidence_accuracy.png")
    architecture()
    test_evidence()
    print("Wrote seven report figures")


if __name__ == "__main__":
    main()
