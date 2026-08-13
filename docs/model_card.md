# Model card

## Model

TF-IDF unigrams and bigrams with logistic regression.

## Training data

Four hundred deterministic synthetic grievances balanced across Ecommerce, Banking, Telecom, Travel, and Appliances. The examples are fictional and contain no customer records.

## Output

The router returns a category and probability when confidence reaches 0.62. Otherwise it returns `Manual review` with the best suggested category.

## Intended use

Demonstrating first-level text routing and abstention in an educational project.

## Prohibited interpretation

The output is not a legal decision, eligibility result, complaint filing, or guarantee that an organisation uses the same categories.

## Limitations

The data is templated, English-only, and narrower than real consumer language. Names, multilingual text, sarcasm, compound grievances, and emerging sectors require additional data and review.

