from __future__ import annotations

import json
from pathlib import Path

from report_template import build_research_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/CivicRoute_Service_Request_Triage_Report.pdf"
FIGURES = ROOT / "reports/figures"


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    sections = [
        {
            "title": "Project overview and problem statement",
            "paragraphs": [
                "CivicRoute reads a written public service request and suggests one of eight service queues. I built it because free-text requests and structured administrative queues do not naturally match, which can create transfers and delay a first response.",
                "The system is an interpretable triage experiment. It does not file a complaint, validate the facts, decide priority or legal rights, or replace an official portal. Low-confidence and very short messages are retained for manual review.",
            ],
        },
        {
            "title": "Source, privacy and preparation",
            "paragraphs": [
                "The source is a 175,784-record Government of India complaint corpus published on Kaggle with an MIT licence and a category mapping workbook. The source system uses the formal word grievance; the project interface uses service request because it is clearer.",
                "I map category codes to organisations, select eight well-represented queues, remove insufficient text, clean repeated redaction markers, and remove exact duplicate request text. The prepared table contains 36,805 rows. Generated text is excluded from Git because de-identified requests can still contain details entered by a citizen.",
            ],
            "table": [
                ["Service queue", "Prepared rows"],
                ["Labour and Employment", "9,458"],
                ["Financial Services", "6,272"],
                ["Direct Taxes", "5,978"],
                ["Agriculture", "5,220"],
                ["Telecommunications", "3,622"],
                ["Postal Services", "3,058"],
                ["Railways", "2,048"],
                ["Road Transport", "1,149"],
            ],
        },
        {
            "title": "Model and evaluation design",
            "paragraphs": [
                "The pipeline uses TF-IDF unigrams and bigrams with class-balanced logistic regression. The saved pipeline contains both text transformation and classification. Predictions below 0.58 confidence are not routed automatically.",
                "A random stratified holdout measures new rows from familiar administrative categories. A harder category-group holdout keeps complete category codes out of training and is the headline generalisation test. This reduces the benefit of category-specific language shared between training and test rows.",
            ],
        },
        {
            "title": "Experiment 1: holdout comparison",
            "figure": FIGURES / "01_holdout_comparison.png",
            "caption": "Figure 1. Macro F1 under random and category-group holdouts.",
            "explanation": [
                [
                    "What I tested",
                    "Whether the model still works when complete administrative category codes are unseen during training.",
                ],
                [
                    "What the graph shows",
                    f"Macro F1 falls from {metrics['macro_f1']:.3f} on the random holdout to {metrics['category_holdout_macro_f1']:.3f} on the harder split.",
                ],
                [
                    "Conclusion",
                    "The random split is optimistic; the category-group result is the more credible headline measure.",
                ],
            ],
        },
        {
            "title": "Experiment 2: confidence review",
            "figure": FIGURES / "02_review_coverage.png",
            "caption": "Figure 2. Automatic routing and manual-review shares on the category-group holdout.",
            "explanation": [
                [
                    "What I tested",
                    "How much work is accepted automatically at the 0.58 confidence threshold.",
                ],
                [
                    "What the graph shows",
                    "84.8 percent is automatically routed and 15.2 percent is retained for review.",
                ],
                [
                    "Conclusion",
                    "The model exposes uncertainty rather than forcing every request into a queue.",
                ],
            ],
        },
        {
            "title": "Experiment 3: queue distribution",
            "figure": FIGURES / "03_queue_distribution.png",
            "caption": "Figure 3. Prepared examples across the eight service queues.",
            "explanation": [
                ["What I tested", "Whether the training labels are balanced across queues."],
                [
                    "What the graph shows",
                    "Labour and Employment has more than eight times the rows of Road Transport.",
                ],
                [
                    "Conclusion",
                    "Class balancing and macro F1 are necessary; accuracy alone would understate minority-queue risk.",
                ],
            ],
        },
        {
            "title": "Experiment 4: class-level F1",
            "figure": FIGURES / "04_class_f1.png",
            "caption": "Figure 4. Random-holdout F1 for each service queue.",
            "explanation": [
                ["What I tested", "Whether one aggregate score hides a weak service queue."],
                [
                    "What the graph shows",
                    "All eight random-holdout F1 values are high, with Agriculture the lowest at 0.988.",
                ],
                [
                    "Conclusion",
                    "Class-level performance is strong on familiar categories, but this does not replace the harder grouped evaluation.",
                ],
            ],
        },
        {
            "title": "Experiment 5: confidence and accuracy",
            "figure": FIGURES / "05_confidence_accuracy.png",
            "caption": "Figure 5. Overall and accepted-route accuracy under both holdouts.",
            "explanation": [
                [
                    "What I tested",
                    "Whether manual review improves the reliability of automatically accepted suggestions.",
                ],
                [
                    "What the graph shows",
                    "Accepted-route accuracy reaches 99.9 percent on the grouped holdout while coverage is 84.8 percent.",
                ],
                [
                    "Conclusion",
                    "The threshold creates a useful accuracy-coverage trade-off, but it cannot detect every confidently wrong request.",
                ],
            ],
        },
        {
            "title": "Results and interpretation",
            "paragraphs": [
                f"The random holdout contains {metrics['test_rows']:,} rows and reaches {metrics['accuracy']:.1%} accuracy with macro F1 {metrics['macro_f1']:.3f}. The category-group holdout contains {metrics['category_holdout_rows']:,} rows from {metrics['category_holdout_categories']} unseen category codes and reaches {metrics['category_holdout_accuracy']:.1%} accuracy with macro F1 {metrics['category_holdout_macro_f1']:.3f}.",
                "At the selected threshold, grouped automatic coverage is 84.8 percent and accepted-route accuracy is 99.9 percent in this snapshot. This result is not a claim about every future organisation, language, emergency request, or taxonomy.",
            ],
        },
        {
            "title": "Limitations, governance and reproducibility",
            "paragraphs": [
                "The corpus is historical and English-dominant. Administrative taxonomies change, similar language can occur across queues, and compound requests may require multiple owners. The system does not detect urgency, malicious content, or sensitive information reliably.",
                "The repository versions acquisition, preparation, both evaluation splits, threshold behaviour, tests, five figures, the model card, and this report. Future work should add time-based validation, independently written and multilingual requests, queue-specific thresholds, drift monitoring, emergency detection, and a secure reviewer interface.",
            ],
        },
        {
            "title": "Conclusion",
            "paragraphs": [
                "CivicRoute demonstrates an interpretable and measurable approach to service-request triage. The central result is not the near-perfect random split; it is the deliberate use of a harder unseen-category split and a visible manual-review path. The project shows how NLP evaluation, class imbalance, confidence thresholds, privacy boundaries, and governance fit together in a student-scale system."
            ],
        },
    ]
    return build_research_report(
        OUTPUT,
        "CivicRoute Service Request Triage",
        "Harika",
        [
            "This report presents an interpretable text-classification project for first-level public service-request triage. I prepared 36,805 labelled and deduplicated requests from 175,784 source records, trained a class-balanced TF-IDF logistic-regression pipeline, and added confidence-based manual review.",
            "A random holdout reached 0.995 macro F1. A harder holdout that excluded complete category codes from training reached 0.974 macro F1 on 12,789 requests. Five experiments examine generalisation, confidence coverage, label imbalance, class-level performance, and accepted-route accuracy.",
        ],
        "text classification; service requests; grouped validation; confidence threshold; human review",
        sections,
    )


if __name__ == "__main__":
    print(build_report())
