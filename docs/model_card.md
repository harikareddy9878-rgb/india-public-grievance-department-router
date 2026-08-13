# Model card

## Model

TF-IDF unigrams and bigrams with class-balanced logistic regression.

## Training data

36,805 labeled and deduplicated complaint texts selected from 175,784 de-identified records in the Government of India grievance corpus. Eight departments are included because they have sufficient labeled examples for a student-scale experiment.

## Output

The router returns a department and probability when confidence reaches 0.58. Otherwise it returns `Manual review` with the best suggestion. Inputs shorter than 25 characters are not automatically routed.

## Evaluation

The fixed stratified held-out set contains 7,361 rows. Overall accuracy and macro F1 are 0.995. At the chosen threshold, 97.5% of test rows are automatically routed and all of those accepted predictions are correct in this snapshot.

## Intended use

Demonstrating interpretable first-level text routing, confidence, and abstention.

## Limitations

Department-specific phrases may make the held-out task easier than genuinely new complaints. The data snapshot does not cover all organisations, future taxonomies, Indian languages, OCR errors, or adversarial inputs. The model does not validate a complaint, select a final officer, or submit information to CPGRAMS.
