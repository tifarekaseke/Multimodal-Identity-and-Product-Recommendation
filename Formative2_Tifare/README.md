# Formative 2: Multimodal ML | Tifare's Section

**Scope:** Data merge, data cleaning, EDA, feature engineering, product recommendation model and evaluation.

This folder is one contributor's module for the full group system. Face recognition, voice verification and overall system integration are **separate group tasks**, not claimed as completed here.

## Reproduce all outputs

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py
python -m unittest discover -s tests -v
python src/predict_product.py --customer-id 178
jupyter notebook notebooks/01_merge_eda_recommendation.ipynb
```

Run from this folder/repository root. Raw CSVs provided in `data/raw/` were exported from the assignment Google Sheets.

## Data sources

- [customer_social_profiles](https://docs.google.com/spreadsheets/d/10up-WdC0a6egYaXLKiMQUotpOvaKZRZvYuWFQx4-RPQ/edit)
- [customer_transactions](https://docs.google.com/spreadsheets/d/1s4WOVm49lmLQ8d9QbbbdgAcRTNh3m0KCH_5ciiaRZw0/edit)

## Data preparation

- Remove complete-row duplicates (social records: 5; transactions: 0).
- Standardize customer IDs (`A178` -> `178`) and numeric/date types.
- Aggregate repeated social profile observations by customer, creating mean engagement, mean interest, number of social observations, number of platforms, dominant platform and sentiment.
- Merge **transactions left to social aggregates**, with `validate='many_to_one'`. All 150 transaction rows preserved; 117 match profiles, 33 do not.
- Keep unmatched transactions in the merged output for transparent validation; use matched rows for training.
- Preserve 10 missing ratings as missing rather than inventing post-purchase values. Ratings and purchase amounts are descriptive-only.

## Leakage-safe features

`engagement_mean, interest_mean, social_activity_count, distinct_platforms, primary_platform, primary_sentiment`

Target: `product_category`. Explicitly exclude customer ID, transaction ID, purchase date, purchase amount and customer rating from predictors. Social profiles have no timestamps, so this model assumes the supplied social profile summary was known prior to the transaction. This cannot be established from the source data and is a limitation.

## Model evaluation (actual run)

- Train rows: **87** from **45** customers.
- Held-out test rows: **30** from **16** previously unseen customers.
- Selected by grouped four-fold macro-F1 CV: **Logistic regression**.
- Test accuracy: **26.7%**; macro F1: **0.150**; weighted F1: **0.206**; multiclass log loss: **2.049**.
- Group splitting prevents any single customer's transactions from appearing in both training and test.

These are **weak results**, close to chance-level performance for five categories. Do not claim strong recommendation performance or deploy this as production authentication/recommendation. More predictive pre-visit data and a larger dataset are needed.

## Repository contents

```text
src/pipeline.py                Full reproducible data + model pipeline
src/predict_product.py         CLI prediction module (for integration after authentication)
notebooks/01_merge_eda_recommendation.ipynb
models/product_recommender.joblib
models/evaluation.json
data/raw/                       Original assignment CSV exports
 data/processed/               Merged data, model data, holdout predictions, quality report
figures/                       Four EDA figures plus test confusion matrix
tests/test_pipeline.py         Merge and leakage checks
report/tifare_section.pdf       Report section for group compilation
```

## Group integration

The group integration member can import/use the saved `models/product_recommender.joblib` after **both** face and voice gates succeed. `src/predict_product.py` is a stand-alone recommendation CLI, not an authentication bypass or full system demonstration. Predictions are for **product categories**, because the source has no individual SKU/product names.

**Privacy:** Do not store or push raw facial photos, biometric embeddings or raw audio into a public repository without informed consent. Configure repository exclusions and use approved private sharing if necessary.
