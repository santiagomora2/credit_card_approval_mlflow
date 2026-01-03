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