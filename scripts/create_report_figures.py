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
    print("Wrote five report figures")


if __name__ == "__main__":
    main()
