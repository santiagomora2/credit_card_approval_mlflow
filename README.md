# Credit Card Approval Prediction: End-to-End MLOps with MLflow

This repository contains an **end-to-end Classic Machine Learning MLOps** mini project, focused on building, tracking, and evaluating credit risk models using **MLflow, scikit-learn**, and reproducible data pipelines.

This project doesn't emphasize model accuracy, but rather **production-grade ML practices**: clean data splits, experiment tracking, class imbalance handling, and evaluation gates.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Pipeline Components](#pipeline-components)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training Pipeline](#2-training-pipeline)
  - [3. Model Selection & Registration](#3-model-selection--registration)
  - [4. Evaluation & Quality Gates](#4-evaluation--quality-gates)
  - [5. Model Promotion](#5-model-promotion)
  - [6. Model Archive](#6-model-archive)
  - [7. Production API & Monitoring](#7-production-api--monitoring)
- [Complete Workflow](#complete-workflow)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project implements a complete MLOps workflow for credit risk assessment, focusing on **production-grade ML engineering** rather than achieving state-of-the-art accuracy. Key features include:

- **Reproducible Data Pipelines**: Stratified splitting with consistent random seeds
- **Experiment Tracking**: MLflow logging for all experiments, metrics, and artifacts
- **Class Imbalance Handling**: Balanced class weights and scale_pos_weight for XGBoost
- **Model Registry**: Centralized model versioning with aliases and promotion workflows
- **Evaluation Gates**: ROC-AUC thresholds for model promotion to staging and production
- **Automated Preprocessing**: Scikit-learn pipelines for consistent feature engineering

---

## Problem Statement

Given applicant demographic and financial information, the goal is to **predict** whether a client is a **good or bad credit risk**.

This is a **highly imbalanced** classification problem, reflecting real-world credit portfolios where defaults are rare but costly.


> **Note on Predictive Power**  
> Models trained **only on demographic data** (age, income, employment) have a known, limited ceiling. In real banking, these features alone typically yield **ROC-AUC scores of 0.50–0.60**—only marginally better than random guessing (AUC=0.5).

> This project uses a classic, demographic-only dataset. The significant predictive lift in real credit scoring comes from **transactional and behavioral data** (e.g., repayment history), which is not used to build predictive features here. Therefore, a model performing within the **0.50–0.60 AUC range is an expected outcome**, not a pipeline failure.

> **The Core Value of This Project**  
> The focus is on demonstrating **production-grade MLOps practices** (tracking, registry, deployment) — the essential infrastructure for when more powerful data sources are integrated.

---

## Dataset

### Source
[Credit Card Approval Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) by rikdifos on Kaggle

### Data Files
- `application_record.csv`: Applicant demographics and financial information
- `credit_record.csv`: Historical credit repayment behavior

### Data Usage Notes

Due to Kaggle's data usage policy:
- Raw data files are **not included** in this repository
- Users must **manually download** the dataset from Kaggle
- Place CSV files in `./data/raw/` directory
- Dataset is used for **educational and research purposes only**

### Target Variable Construction

Since the dataset lacks an explicit approval outcome, a **proxy target variable** is constructed using historical credit repayment behavior:

**Labeling Rules:**
- **BAD (0)**: Clients with any repayment status ≥ 3 (90+ days past due)
- **GOOD (1)**: Clients with all repayment statuses in {X, C, 0, 1, 2}
- **EXCLUDED**: Clients with fewer than 3 non-X status records (insufficient data)

**To prevent label leakage:**
- Credit history is used **exclusively** for target construction
- Only application and demographic data are used as model features
- Applicants without credit history are excluded from training

---

## Project Architecture

```
credit_card_approval_mlflow/
├── data/
│   ├── raw/                    # Original Kaggle data (not committed)
│   ├── processed/              # Preprocessed application & credit records
│   ├── features/               # Final train/val/test splits
│   └── production/             # Production prediction data
├── src/
│   ├── config/
│   │   └── config.yaml         # Centralized configuration
│   ├── schemas/
│   │   ├── input_schema.json   # Model input schema
│   │   └── output_schema.json  # Model output schema
│   ├── data/
│   │   ├── preprocessing.py    # Raw data preprocessing
│   │   ├── features.py         # Feature engineering & target creation
│   │   └── split_data.py       # Stratified train/val/test splitting
│   ├── models/
│   │   ├── train.py            # Model training with MLflow
│   │   ├── select_and_register.py  # Best model selection & registration
│   │   ├── evaluate.py         # Model evaluation on test set
│   │   ├── promote.py          # Model promotion between environments
│   │   └── archival.py         # Model archival & unarchival
│   ├── serving/
│   │   └── app.py              # FastAPI production backend
│   └── monitoring/
│       ├── log_predictions.py  # Production prediction logging
│       └── drift.py            # Drift detection and monitoring (in development)
├── environment.yml             # Conda environment file
└── README.md
```

---

## Pipeline Components

### 1. Data Preprocessing

**Files**: `preprocessing.py`, `features.py`, `split_data.py`

#### Stage 1: Raw Data Preprocessing (`preprocessing.py`)

**Objective**: Clean and transform raw application and credit records

**Operations**:
1. **ID Alignment**: Filter to keep only IDs present in both datasets
2. **Binary Encoding**: Convert `FLAG_OWN_CAR` and `FLAG_OWN_REALTY` from Y/N to 1/0
3. **Age Calculation**: Transform `DAYS_BIRTH` (days before current) to `AGE` in years
4. **Employment Duration**: Convert `DAYS_EMPLOYED` to `YEARS_EMPLOYED`
   - Handles unemployed cases (positive values → 0)
5. **Missing Values**: Fill null `OCCUPATION_TYPE` with "None"

**Output**: `application_record_processed.csv`, `credit_record_processed.csv`

#### Stage 2: Feature Engineering & Target Creation (`features.py`)

**Objective**: Create final feature set and target variable

**Feature Selection**:
- **Dropped**: `CNT_CHILDREN`, `FLAG_MOBIL`, `DAYS_BIRTH`, `DAYS_EMPLOYED` (redundant or non-informative)
- **Retained**: All other demographic and financial features

**Target Construction**:
```python
def classify_client(statuses):
    # Exclude if < 3 non-X status values (insufficient data)
    # BAD (0) if any STATUS in {3, 4, 5} (90+ days past due)
    # GOOD (1) if all STATUS in {X, C, 0, 1, 2}
```

**Duplicate Handling**:
- Sorts by TARGET (ascending) to prioritize BAD labels
- Drops duplicates on feature columns, keeping first occurrence
- **Rationale**: Conservative approach prioritizes identifying risky clients

**Output**: `data.csv` with features and target

#### Stage 3: Data Splitting (`split_data.py`)

**Objective**: Create stratified train/validation/test splits, drop `ID` column (not needed anymore)

**Method**: Two-stage `StratifiedShuffleSplit`
1. **Stage 1**: Train (70%) vs. Temp (30%)
2. **Stage 2**: Split Temp into Val (15%) and Test (15%)

**Key Features**:
- **Stratification**: Maintains class distribution across splits
- **Reproducibility**: Fixed random seed for consistent splits
- **Configuration**: Split ratios defined in `config.yaml`

**Output**: `train.csv`, `val.csv`, `test.csv`

---

### 2. Training Pipeline

**File**: `train.py`

#### Preprocessing Pipeline

All models use a **scikit-learn `Pipeline`** with a `ColumnTransformer`:

```python
ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
```

**Benefits**:
- **Consistency**: Same preprocessing applied during training and inference
- **No Data Leakage**: Transformers fit only on training data
- **Production-Ready**: Pipeline can be serialized and deployed as a single object

#### Models Implemented

| Model | Imbalance Handling | Hyperparameters |
|-------|-------------------|-----------------|
| **Logistic Regression** | `class_weight="balanced"` | C, max_iter |
| **Random Forest** | `class_weight="balanced_subsample"` | n_estimators, max_depth, min_samples_leaf |
| **XGBoost** | `scale_pos_weight` (computed from data) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_lambda, reg_alpha |
| **Majority Baseline** | N/A | Predicts class 1 (GOOD) for all samples |
| **Random Baseline** | N/A | Random predictions |

#### Class Imbalance Strategy

Credit datasets are inherently imbalanced (few BAD cases, many GOOD cases). This project addresses imbalance through:

1. **Balanced Class Weights**: Automatically adjust loss function to penalize minority class errors more heavily
2. **XGBoost Scale Pos Weight**: Computed as `(# negative samples) / (# positive samples)`
3. **ROC-AUC Metric**: Threshold-agnostic metric suitable for imbalanced datasets

#### MLflow Tracking

Each training run logs:
- **Parameters**: Model type, hyperparameters, baseline flag
- **Metrics**: 
  - `val_roc_auc`: Validation ROC-AUC score (primary metric)
  - `train_bad_rate`: Proportion of BAD clients in training set
  - `val_bad_rate`: Proportion of BAD clients in validation set
- **Artifacts**:
  - Confusion matrix plot
  - ROC curve plot
  - Serialized model pipeline

#### Running Training

```bash
# Train XGBoost model
python3 ./src/models/train.py --model xgboost

# Train Random Forest
python3 ./src/models/train.py --model random_forest

# Train Logistic Regression
python3 ./src/models/train.py --model logistic_regression

# Train baselines
python3 ./src/models/train.py --model majority_baseline
python3 ./src/models/train.py --model random_baseline
```

**MLflow UI**: View experiments at `http://127.0.0.1:5000`

---

### 3. Model Selection & Registration

**File**: `select_and_register.py`

#### Selection Process

1. **Query Experiments**: Search all completed runs in MLflow
2. **Ranking**: Order by `val_roc_auc` descending
3. **Best Model**: Select top-ranked run
4. **Test Evaluation**: Evaluate best model on held-out test set

#### Registration Criteria

A model is registered **only if**:
```python
test_roc_auc >= threshold  # Defined in config.yaml
```

#### Registration Workflow

If the model passes the threshold:
1. **Register Model**: Create new version in MLflow Model Registry
2. **Assign Alias**: Tag as `@candidate` for staging evaluation
3. **Log Schemas**: Attach input/output schemas for validation
4. **Log Metadata**: Record evaluation metrics and approval status

If the model fails:
- Log rejection status
- Do not register model
- Allows for model retraining or hyperparameter tuning

#### Running Selection

```bash
python3 ./src/models/select_and_register.py
```

**Output**:
```
Best run: <run_id>
Validation ROC-AUC: 0.XXXX
Test ROC-AUC: 0.XXXX
Model registered as version X and aliased as @candidate
```

---

### 4. Evaluation & Quality Gates

**File**: `evaluate.py`

#### Staging Evaluation

Before promoting to production, the candidate model undergoes a **staging evaluation**:

1. **Load Model**: Retrieve model from registry by URI
2. **Test Set Evaluation**: Evaluate on held-out test data
3. **Quality Gate**: Check if `test_roc_auc >= threshold`
4. **Logging**: Record evaluation in MLflow with staging tag

#### Evaluation Metric: ROC-AUC

**ROC-AUC (Area Under the Receiver Operating Characteristic Curve)** is chosen because:
- **Threshold-Agnostic**: Evaluates model across all classification thresholds
- **Imbalance-Robust**: Not biased by class imbalance
- **Business-Aligned**: Balances true positive rate (approving good clients) vs. false positive rate (approving bad clients)

#### Running Evaluation

```bash
# Evaluate candidate model using alias
python3 ./src/models/evaluate.py \
  --model-uri "models:/credit_card_approval_model@candidate"
```

**Output**:
```
Test ROC-AUC (registry model): 0.XXXX
Model PASSED staging evaluation.
```

---

### 5. Model Promotion

**File**: `promote.py`

#### Promotion Workflow

Models progress through environments using **MLflow Model Registry Aliases**:

```
Training Runs → @candidate → Staging → @champion (Production)
```

#### Promotion Steps

**1. Model is selected as best performer**

**2. Model passes staging evaluation**

**3. Promote Candidate to Production (assign champion alias):**:
```bash
# Add champion alias to the approved version (this becomes the production model)
python3 ./src/models/promote.py \
  --model-name credit_card_approval_model \
  --version 1 \
  --alias champion \
  --action add

# Optional: Remove candidate alias if no longer needed
python3 ./src/models/promote.py \
  --model-name credit_card_approval_model \
  --version 1 \
  --alias candidate \
  --action remove
```

#### Alias Management vs. Deprecated Staging

**Alias Management Benefits**

This workflow uses MLflow aliases instead of deprecated stages:

- **Mutable**: Aliases can be reassigned to different versions without changing consumer code
- **Auditable**: Clear history of which version has which alias at any time
- **Flexible**: Multiple aliases can point to the same model version
- **Standard Practices**:

    - `champion`: Primary production model (auto-prefixed as `@champion`)
    - `candidate`: New candidate ready for evaluation (auto-prefixed as `@candidate`)
    - `canary`: Model for canary testing (auto-prefixed as `@canary`)

### 6. Model Archive

**File**: `archive_model.py`

#### Archival Approach

With MLflow's deprecated stages, archival is handled using **tags** instead of a dedicated "Archived" stage:

1. **Tag-Based Archival**: Models are marked with `archived=true` tag
2. **Audit Trail**: `archived_at` timestamp and `archival_reason` tags provide history
3. **Clean Registry**: Active aliases are removed during archival

#### Archival Workflow

**Archive a model version**:
```bash
python3 src/models/archive_model.py archive \
  --model-name credit_card_approval_model \
  --version 2 \
  --reason "Replaced by version 3 with improved ROC-AUC"
```

**What happens:**

- Removes all active aliases (e.g., `@champion`, `@candidate`)
- Sets `archived=true` tag
- Sets `archived_at` timestamp (ISO format)
- Sets `archival_reason` tag (if provided)

**Unarchive a model version**

```bash
python3 src/models/archive_model.py unarchive \
  --model-name credit_card_approval_model \
  --version 2 \
  --alias candidate
```

**What happens:**
- Removes archival tags from a model version 
- Adds an optional alias

**Note:** Archived models are still accessible by version number, just not through aliases:

```bash
# Serve archived version directly (by version number)
mlflow models serve \
  -m "models:/credit_card_approval_model/2" \
  -p 5002 \
  --env-manager local
```
---

### 7. Production API & Monitoring

Files: `serving/app.py`, `monitoring/log_predictions.py`, `monitoring/drift.py`

#### Production API (`serving/app.py`)

**Objective:** FastAPI backend for production model serving with request logging

**Features:**

- **REST API Endpoints**: /predict for single/batch predictions, /health for monitoring
- **CORS Support**: Configurable cross-origin resource sharing
- **Latency Tracking**: Measures end-to-end prediction time
- **Error Handling**: Structured error responses and logging

**Running the API:**

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

#### Prediction Logging (`monitoring/log_predictions.py`)

**Objective:** Log all production predictions for monitoring and drift detection

**Logging Strategy:**

- **CSV Storage**: Append all predictions with metadata to `data/production/production_data.csv`
- **MLflow Metrics**: Log aggregated metrics (mean, std, latency) to `production_monitoring` experiment
- **Single Function**: `predict_and_log()` handles model loading, prediction, and logging in one call

**Logged Data:**

- Input features
- Prediction probabilities and binary classifications
- Timestamps and model version
- Prediction latency

**Architecture**:
```
FastAPI → MLflow Model → Prediction
   ↓
MLflow Tracking (log requests, responses, latencies)
```

#### Drift Monitoring (`monitoring/drift.py`)

**Objective:** Monitor model performance and data drift over time

(Currently in development)

## Complete Workflow

### Prerequisites

1. **Download Data**: Obtain dataset from Kaggle and place in `./data/raw/`

2. **Set Up Environment**
Create and activate the Conda environment from the included specification:

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate cca_mlflow
```

### Full Pipeline Execution

#### Step 1: Data Preparation

```bash
python3 ./src/data/split_data.py
```

**What this does**:
1. Runs `preprocess_data()`: Cleans raw CSVs
2. Runs `extract_features_and_label()`: Creates target variable
3. Performs stratified train/val/test split (70/15/15)

**Output**: `train.csv`, `val.csv`, `test.csv` in `./data/features/`

---

#### Step 2: Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

**Access MLflow UI**: Navigate to `http://127.0.0.1:5000`

---

#### Step 3: Train Models

```bash
# Train all models
python3 ./src/models/train.py --model majority_baseline
python3 ./src/models/train.py --model random_baseline
python3 ./src/models/train.py --model logistic_regression
python3 ./src/models/train.py --model random_forest
python3 ./src/models/train.py --model xgboost
```

**Monitor**: View experiment runs in MLflow UI

---

#### Step 4: Select & Register Best Model

```bash
python3 ./src/models/select_and_register.py
```

**What this does**:
1. Identifies run with highest validation ROC-AUC
2. Evaluates on test set
3. If `test_roc_auc >= threshold`: Registers model with `@candidate` alias
4. If failed: Logs rejection (iterate on training)

---

#### Step 5: Staging Evaluation

```bash
python3 ./src/models/evaluate.py \
 --model-uri "models:/credit_card_approval_model@candidate"
```

**Quality Gate**: Must pass ROC-AUC threshold to proceed

---

#### Step 7: Promote to Production

```bash
# Add @champion alias to the approved version (this becomes the production model)
python3 ./src/models/promote.py \
 --model-name credit_card_approval_model \
 --version 1 \
 --alias champion \
 --action add

# Optional: Remove @candidate alias if no longer needed
python3 ./src/models/promote.py \
 --model-name credit_card_approval_model \
 --version 1 \
 --alias candidate \
 --action remove
```

---

#### Step 8: Model Serving

```bash
# Serve using the @champion alias (production model)
mlflow models serve \
  -m models:/credit_card_approval_model@champion \
  -p 5001 \
  --env-manager local
```

**Endpoint**: Model accessible at `http://127.0.0.1:5001/invocations`

---

#### Step 9: Test Inference

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
        "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "AGE", "YEARS_EMPLOYED"
      ],
      "data": [
        ["M", 1, 0, 150000, "Working", "Higher education", "Married",
         "House / apartment", 1, 1, 0, "IT staff", 2, 35, 5]
      ]
    }
  }'
```

**Expected Response**: Probability score between 0 and 1

#### Step 10: Deploy Production API

```bash
# Start FastAPI backend
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

#### Step 11: Test Production API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 0,
    "AMT_INCOME_TOTAL": 150000,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "FLAG_WORK_PHONE": 1,
    "FLAG_PHONE": 1,
    "FLAG_EMAIL": 0,
    "OCCUPATION_TYPE": "IT staff",
    "CNT_FAM_MEMBERS": 2,
    "AGE": 35,
    "YEARS_EMPLOYED": 5
  }]'
```

#### Step 12: Monitor Production Data

- CSV Logs: Check `data/production/production_data.csv` for all predictions
- MLflow Metrics: View aggregated metrics in `production_monitoring` experiment
- Drift Detection: Periodically run drift checks to monitor model performance

### Additional Useful Commands

```bash
# To check which version has which alias:
mlflow models list-alias-versions --name credit_card_approval_model

# To serve a model with custom alias:
mlflow models serve -m "models:/credit_card_approval_model@canary" -p 5002

# To serve without any alias (specific version):
mlflow models serve -m "models:/credit_card_approval_model/1" -p 5003
```
---

## Model Performance

### Evaluation Metrics

- **Primary Metric**: ROC-AUC (threshold-agnostic, imbalance-robust)
- **Secondary Metrics**: Confusion matrix, precision-recall at various thresholds

### Baseline Comparison

| Model | Validation ROC-AUC |
|-------|-------------------|
| Random Baseline | 0.5218 |
| Majority Baseline | 0.50 |
| Logistic Regression | 0.5154 |
| Random Forest | 0.5355 |
| **XGBoost** | **0.5885** |

> Note: As mentioned in the introduction, this project **doesn't emphasize model accuracy**, but rather **production-grade ML practices**

### Model Interpretation

- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, false negatives
- **ROC Curve**: Shows trade-off between sensitivity and specificity across thresholds
- **Feature Importance** (for tree-based models): Identifies most predictive features

---

## Future Enhancements

### 1. Model Monitoring

- **Data Drift Detection**: Monitor feature distributions over time
- **Prediction Drift**: Track changes in prediction distributions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) by rikdifos on Kaggle
- **MLflow**: Open-source platform for the ML lifecycle
- **Scikit-learn & XGBoost**: Core machine learning libraries

---

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ to learn production-grade ML engineering**