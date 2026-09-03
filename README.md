# Olist Late Delivery Prediction - MLOps Project

End-to-end MLOps project using the Brazilian E-Commerce Public Dataset by Olist.

## Objective

Predict whether an e-commerce order will be delivered late while preventing data leakage and evaluating performance honestly on future chronological data.

## Task 1 - Relational Data Foundation

Completed:

- PostgreSQL 16 with Docker Compose
- nine Olist relational tables
- SQL schema creation and bulk ingestion
- data-integrity validation
- Python/PostgreSQL integration
- order-level relational dataset construction

## Task 2 - Notebook ML Workflow

Completed with six notebooks:

1. `01_read_and_join.ipynb` - build the order-level ML table
2. `02_create_labels.ipynb` - create the late-delivery target
3. `03_train_validation_test_split.ipynb` - chronological split
4. `04_train_eda.ipynb` - training-only EDA
5. `05_feature_engineering.ipynb` - feature engineering and fitted preprocessing
6. `06_train_tune_evaluate.ipynb` - baseline, validation tuning, and final test

## Chronological Split

- Train: 67,529 rows
- Validation: 14,470 rows
- Test: 14,471 rows
- Cross-split overlap: 0

Detailed EDA is performed on training data only.

## EDA Controls

Notebook 4 includes:

- missing-value analysis
- target imbalance
- numerical distributions
- IQR outlier audit
- rare and high-cardinality categories
- categorical crosstabs
- geographic analysis
- temporal and seasonality analysis
- leakage auditing

No holiday feature is invented because the source dataset contains no authoritative Brazilian holiday calendar.

## Feature Engineering

Notebook 5 starts with 51 prediction-time predictors and produces 184 encoded model features.

The fitted preprocessing pipeline uses:

- median numerical imputation
- missing-value indicators
- standardization
- categorical imputation
- one-hot encoding
- infrequent-category handling

The preprocessor is fitted on training data only.

Validation and test use the same saved fitted object with `.transform()` and never refit preprocessing.

## Leakage Prevention

Excluded predictors include:

- delivery outcome timestamps
- `delivery_delay_days`
- post-delivery review information
- `is_late`
- order, customer, and seller identifiers

## Model Selection

Notebook 6 begins with a dummy baseline.

Selected model:

- Balanced Logistic Regression
- C = 0.1
- Threshold = 0.692968

Validation:

- Baseline PR-AUC: 0.043124
- Model PR-AUC: 0.134065
- ROC-AUC: 0.775557

## Final Test Results

The chronological test set is opened only after model and threshold selection are frozen.

- Test rows: 14,471
- Late orders: 620
- PR-AUC: 0.081893
- ROC-AUC: 0.657710
- Precision: 0.071931
- Recall: 0.280645
- F1: 0.114511
- PR-AUC lift: 1.91x

The weaker final test performance is retained rather than used for further tuning.

## Reproducibility

Run notebooks in order:

`01 -> 02 -> 03 -> 04 -> 05 -> 06`

Generated Parquet, JSON, transformer, and model artifacts remain local and are excluded from Git.

## Next Stage

Task 2 remains notebook-based.

Production scripts, automated testing, CI/CD, deployment, and monitoring belong to later MLOps tasks.
