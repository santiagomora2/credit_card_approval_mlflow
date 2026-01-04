# Credit Card Approval Prediction: End-to-end with MLFlow

This repository contains an **end-to-end Classic Machine Learning MLOps** mini project, focused on building, tracking, and evaluating credit risk models using **MLflow, scikit-learn**, and reproducible data pipelines.

This project doesn't emphasize model accuracy, but rather **production-grade ML practices**: clean data splits, experiment tracking, class imbalance handling, and evaluation gates.

---

## Problem Statement

Given applicant demographic and financial information, the goal is to **predict** whether a client is a **good or bad credit risk**.

This is a **highly imbalanced** classification problem, reflecting real-world credit portfolios where defaults are rare but costly.

## Dataset

**Source**:

Credit Card Approval Prediction Dataset, provided by *rikdifos* on [Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

**Data Usage Notes**

* Due to Kaggle’s data usage policy, raw data files are **not included** in this repository.

* Users must download the dataset **manually** and place the CSV files in the `data/raw/` directory.

* Processed datasets (train/val/test) are generated **locally**.

* Dataset is used for **educational and research purposes**.

## Target Definition

Since the dataset does not provide an explicit approval outcome, a **proxy target variable** was constructed using historical credit repayment behavior.

Clients were labeled as bad if they exhibited any **repayment status of 90 days past due** or worse (STATUS ≥ 3). Clients without severe delinquency were labeled as good.

To avoid label leakage, **credit history data** was used **exclusively for target construction** and was not included as input features. Only application and demographic data were used for model training.

Applicants without credit history were **excluded from model training**, as no repayment behavior was available to derive the target label.

## Full run

download and place raw data in `./data/raw`

Clean, process and  split data
```shell
python3 ./src/data/split_data.py
```

Start MLflow ui server
```shell
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Train models
```shell
python3 ./src/models/train.py --model xgboost (and other baselines/models)
```

Select and register best model
```shell
python3 ./src/models/select_and_register.py
```

Promote to staging

```shell
python3 src/models/promote.py \
    --src-name credit_card_approval_model \
    --src-alias candidate \
    --dst-name credit_card_approval_model_staging
```

Evaluate

```shell
python src/models/evaluate.py \
  --model-uri models:/credit_card_approval_model_staging/latest

```

Promote to production

```shell
python src/models/promote.py \
  --src-name credit_card_approval_model_staging \
  --src-alias latest \
  --dst-name credit_card_approval_model_production
```

Serve (MLflow)

```shell
mlflow models serve \
  -m models:/credit_card_approval_model_production/latest \
  -p 5001 \
  --env-manager local
```

Test curl

```shell
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

Future (developing): FastAPI backend for serving models and handling API requests. This will enable data logging from requests to mlflow and model monitoring.