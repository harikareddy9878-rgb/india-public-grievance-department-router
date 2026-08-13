# Consumer Grievance Router India

This applied AI project classifies a short consumer grievance into one of five service sectors and abstains when its confidence is too low. The result is an interpretable routing aid, not an automated complaint decision.

![Routing evaluation](evidence/routing_evaluation.png)

## Problem

People describe similar service problems with different words. A complaint about an unauthorised bank charge, a delayed online refund, or a cancelled train should reach a relevant first-level queue without forcing the user to understand an internal taxonomy.

## Root cause addressed

Keyword-only rules are brittle and confident automation can misroute ambiguous text. This project combines TF-IDF language features with logistic regression, exposes class probabilities, and sends low-confidence cases to manual review.

## Purpose

The project demonstrates a small, testable natural language routing workflow using synthetic Indian consumer grievance examples. It does not submit complaints, determine legal rights, or replace the National Consumer Helpline.

## Categories

`Ecommerce`, `Banking`, `Telecom`, `Travel`, and `Appliances`.

## Repository guide

| Folder | Contents |
| --- | --- |
| `data` | Synthetic training examples and evaluation summary |
| `src` | Dataset generation, model training, and routing code |
| `models` | Saved text classification pipeline |
| `evidence` | Confusion matrix and confidence analysis |
| `reports` | Detailed PDF project report |
| `tests` | Routing, abstention, and input safety checks |

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python scripts/build_report.py
pytest
ruff check src scripts tests
```

For official help, use the [National Consumer Helpline](https://consumerhelpline.gov.in/). This repository is an educational project and is not affiliated with the helpline.

## Author

Harika

