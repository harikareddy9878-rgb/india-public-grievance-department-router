# India Public Grievance Department Router

This applied NLP project prepares a de-identified Government of India grievance corpus and routes a written grievance to one of eight departments. The model exposes confidence and sends uncertain cases to manual review.

![Routing evaluation](evidence/routing_evaluation.png)

## Project question

Can a compact, interpretable language model route real public grievance text while retaining a visible review path for uncertain cases?

## Result

| Measure | Result |
| --- | ---: |
| Source grievance records | 175,784 |
| Labeled model rows after selection and deduplication | 36,805 |
| Departments | 8 |
| Held out test rows | 7,361 |
| Accuracy | 99.5% |
| Macro F1 | 0.995 |
| Automatic coverage at 0.58 confidence | 97.5% |
| Accuracy among automatically routed cases | 100.0% |

The high score reflects strong department-specific language and category text within this snapshot. It should not be read as proof that the model can route every future grievance, language, department, or compound case.

## Data source

The acquisition script downloads the [Government of India: Grievance report](https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report) corpus. The Kaggle page states an MIT licence. The source contains files explicitly named `no_pii_grievance.json` and a category mapping workbook.

This project uses the complaint text and mapped department only. It does not attempt re-identification, does not publish complainant attributes, and does not make a decision on the validity of a grievance.

## Departments

Agriculture, Direct Taxes, Financial Services, Labour and Employment, Postal Services, Railways, Road Transport, and Telecommunications.

## Modelling approach

The pipeline removes repeated redaction markers and whitespace noise, preserves the complaint wording, creates TF-IDF unigram and bigram features, and trains a class-balanced logistic regression model. The saved pipeline contains both text transformation and classification.

The 0.58 review threshold is evaluated on the held-out set. Short messages are rejected before prediction because they do not provide enough context for reliable routing.

## Intended boundary

The output is a first-level routing suggestion. It does not lodge a grievance, decide eligibility, determine legal rights, replace CPGRAMS, or guarantee that a department currently uses the same categories. Multilingual, abusive, personally identifying, and compound submissions need additional safeguards and human handling.

## Repository guide

| Path | Contents |
| --- | --- |
| `scripts/acquire_data.py` | Corpus download, category mapping, text preparation, and audit summary |
| `src/train.py` | Model training, threshold evaluation, and evidence generation |
| `src/route.py` | Inference and manual review behaviour |
| `data/processed` | Prepared training table and evaluation JSON |
| `docs/model_card.md` | Intended use, limitations, and evaluation notes |
| `reports` | Detailed PDF project report |
| `tests` | Text cleaning, routing, and abstention checks |

[Read the ten page project report](reports/India_Public_Grievance_Department_Router_Report.pdf)

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

For official grievance services, use [CPGRAMS](https://pgportal.gov.in/). This repository is an independent educational project.

## Author

Harika
