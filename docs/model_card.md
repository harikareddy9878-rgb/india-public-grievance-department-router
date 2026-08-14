# CivicRoute model card

## Model

TF-IDF unigrams and bigrams with class-balanced logistic regression.

## Training data

36,805 labelled and deduplicated complaint texts selected from 175,784 de-identified source records. Eight service queues have enough labelled examples for this project.

## Output

The model returns a service queue and probability when confidence reaches 0.58. Otherwise it returns `Manual review` with the best suggestion. Inputs shorter than 25 characters are not automatically routed.

## Evaluation

The random stratified holdout contains 7,361 rows and measures requests drawn from familiar categories. The category-group holdout contains 12,789 rows from 90 category codes excluded from training. This second split is the primary generalisation check because it reduces the benefit of category-specific wording shared across training and test data.

Random holdout macro F1 is 0.995. Category-group holdout macro F1 is 0.974. At the chosen confidence threshold, the harder split automatically routes 84.8% of rows and reaches 99.9% accuracy among accepted suggestions in this snapshot.

## Intended use

Demonstrating interpretable first-level service-request triage, confidence, grouped evaluation, and abstention.

## Limitations

The dataset uses historic administrative categories and does not cover every organisation, future taxonomy, Indian language, OCR error, emergency request, or adversarial input. The system does not validate a complaint, determine priority, select a final officer, or submit information to a government platform.
