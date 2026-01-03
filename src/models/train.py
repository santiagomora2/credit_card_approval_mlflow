import argparse
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import mlflow.xgboost
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Utils

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_confusion_and_roc(y_true, y_probs, model_name: str):
    """Log confusion matrix and ROC curve as MLflow artifacts."""
    y_preds = (y_probs >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["BAD", "GOOD"])

    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm, values_format="d")
    ax_cm.set_title(f"{model_name} - Validation Confusion Matrix")

    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    plt.close(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true,
        y_probs,
        ax=ax_roc,
        name=model_name
    )
    ax_roc.set_title(f"{model_name} - Validation ROC Curve")

    mlflow.log_figure(fig_roc, "roc_curve.png")
    plt.close(fig_roc)

# Main

def main(model_type: str):
    config = load_config("src/config/config.yaml")

    # MLflow setup

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    mlflow.sklearn.autolog(log_models=True)
    mlflow.xgboost.autolog(log_models=True)

    # Load data

    target_col = config["data"]["target_column"]

    train_df = pd.read_csv(config["data"]["train_path"])
    val_df = pd.read_csv(config["data"]["val_path"])

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    rng = np.random.default_rng(config["split"]["random_state"])

    # Model selection

    model = None
    is_baseline = False

    if model_type == "logistic_regression":
        cfg = config["model"]["logistic_regression"]
        model = LogisticRegression(
            max_iter=cfg["max_iter"],
            C=cfg["C"],
            class_weight="balanced",
            n_jobs=-1
        )

    elif model_type == "random_forest":
        cfg = config["model"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf = cfg["min_samples_leaf"], 
            class_weight="balanced_subsample",
            random_state=config["split"]["random_state"],
            n_jobs=-1
        )

    elif model_type == "xgboost":
        cfg = config["model"]["xgboost"]

        model = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            reg_alpha=cfg["reg_alpha"],
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=(y_train == 1).sum() / (y_train == 0).sum(),
            random_state=config["split"]["random_state"],
            n_jobs=-1
        )


    elif model_type == "majority_baseline":
        is_baseline = True

    elif model_type == "random_baseline":
        is_baseline = True

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Training & Evaluation

    with mlflow.start_run(run_name=model_type):

        # Train or simulate
        if not is_baseline:
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]

        else:
            if model_type == "majority_baseline":
                val_probs = np.ones(len(y_val))

            elif model_type == "random_baseline":
                val_probs = rng.random(len(y_val))

        # Metrics
        val_roc_auc = roc_auc_score(y_val, val_probs)

        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.log_metric("train_bad_rate", (y_train == 0).mean())
        mlflow.log_metric("val_bad_rate", (y_val == 0).mean())
        mlflow.log_param("is_baseline", is_baseline)

        print(f"[DONE] -{model_type}- Validation ROC-AUC: {val_roc_auc:.4f}")

        # Artifacts
        log_confusion_and_roc(y_val, val_probs, model_type)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit approval models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "logistic_regression",
            "random_forest",
            "xgboost",
            "majority_baseline",
            "random_baseline",
        ],
        help="Model type to train"
    )

    args = parser.parse_args()
    main(args.model)
