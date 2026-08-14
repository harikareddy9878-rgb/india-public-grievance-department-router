# CivicRoute Service Request Triage

CivicRoute is an applied NLP project that reads a written public service request and suggests one of eight service queues. It shows a confidence score and sends uncertain requests to manual review instead of forcing every message into a category.

![CivicRoute evaluation](evidence/routing_evaluation.png)

## Why this project

Citizens describe delayed refunds, pension problems, postal issues, train services, tax matters, telecom faults, road transport, and agriculture services in free text. Service teams work with structured queues. The mismatch can create avoidable transfers and slow first responses.

CivicRoute tests whether an interpretable language model can provide a useful first routing suggestion while keeping human review visible.

## Result

| Measure | Result |
| --- | ---: |
| Source service requests | 175,784 |
| Prepared and deduplicated model rows | 36,805 |
| Service queues | 8 |
| Random held-out rows | 7,361 |
| Random holdout macro F1 | 0.995 |
| Category-group holdout rows | 12,789 |
| Category-group holdout macro F1 | 0.974 |
| Automatic coverage on the harder holdout | 84.8% |
| Accepted-route accuracy on the harder holdout | 99.9% |

The category-group holdout is the more realistic headline result. It keeps entire complaint categories out of training, so the model must handle requests from unseen category codes instead of only new rows from familiar categories.

## Service queues

Agriculture, direct taxes, financial services, labour and employment, postal services, railways, road transport, and telecommunications.

## Data and privacy

The acquisition script downloads the [Government of India complaint corpus](https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report). The Kaggle page calls it a grievance report; in this repository, `service request` is used in the interface because it is clearer. The original source terminology remains only where provenance requires it.

The dataset page states an MIT licence and includes a file named `no_pii_grievance.json` with a category mapping workbook. CivicRoute uses complaint text and mapped service ownership only. It does not attempt re-identification or publish the prepared text table in Git.

## Modelling and evaluation

The pipeline cleans repeated redaction markers and whitespace, creates TF-IDF unigram and bigram features, and trains class-balanced logistic regression.

Two evaluations are reported:

1. A stratified random holdout measures performance on new requests from the known data distribution.
2. A category-group holdout keeps complete category codes out of training and exposes how well the model generalises to less familiar request types.

Messages below the confidence threshold are returned as `Manual review`. Very short inputs are rejected before prediction.

## Boundaries

CivicRoute is a first-level triage experiment. It does not file a complaint, decide whether a request is valid, determine legal rights, replace CPGRAMS, or select a final officer. Personally identifying, multilingual, abusive, emergency, and multi-topic requests need additional controls and human handling.

## Repository guide

| Path | Contents |
| --- | --- |
| `scripts/acquire_data.py` | Source download, category mapping, preparation, and audit summary |
| `src/train.py` | Random and category-group evaluation, model training, and evidence |
| `src/route.py` | Confidence-aware service queue inference |
| `data/processed/evaluation.json` | Reproducible evaluation results |
| `docs/model_card.md` | Intended use, evaluation design, and limitations |
| `reports` | Ten-page PDF project report |
| `tests` | Cleaning, routing, and manual-review checks |

[Read the research-style project report](reports/CivicRoute_Service_Request_Triage_Report.pdf)

The report is a plain research-style document with an abstract, model design, five explained evaluation figures, limitations, governance notes, and conclusion.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/acquire_data.py
python src/train.py
python scripts/build_report.py
pytest
ruff check src scripts tests
```

For an official public complaint, use [CPGRAMS](https://pgportal.gov.in/). This repository is an independent educational project.

## Author

Harika
