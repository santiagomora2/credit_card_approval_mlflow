import yaml
import argparse
import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(model_uri: str):
    config = load_config("src/config/config.yaml")
    threshold = config["evaluation"]["threshold"]

    test_df = pd.read_csv(config["data"]["test_path"])
    target_col = config["data"]["target_column"]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    model = mlflow.pyfunc.load_model(model_uri)

    test_probs = model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, test_probs)

    print(f"Test ROC-AUC (registry model): {test_roc_auc:.4f}")

    with mlflow.start_run(run_name="staging_evaluation"):
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.set_tag("evaluation_stage", "staging")

        if test_roc_auc >= threshold:
            mlflow.set_tag("promotion_status", "approved")
            print("Model PASSED staging evaluation.")
        else:
            mlflow.set_tag("promotion_status", "rejected")
            raise RuntimeError("Model failed staging gate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    args = parser.parse_args()

    main(args.model_uri)
