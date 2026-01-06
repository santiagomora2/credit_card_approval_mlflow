import yaml
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    # Load config and schema paths
    config = load_config("src/config/config.yaml")

    experiment_name = config["mlflow"]["experiment_name"]
    tracking_uri = config["mlflow"]["tracking_uri"]
    threshold = config["evaluation"]["threshold"]
    model_name = config["mlflow"]["model_name"]

    input_schema_path = Path("src/schemas/input_schema.json")
    output_schema_path = Path("src/schemas/output_schema.json")

    # Load test data
    test_df = pd.read_csv(config["data"]["test_path"])
    target_col = config["data"]["target_column"]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError("MLflow experiment not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_roc_auc DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No completed runs found.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    val_roc_auc = best_run.data.metrics["val_roc_auc"]

    print(f"Best run: {run_id}")
    print(f"Validation ROC-AUC: {val_roc_auc:.4f}")

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    test_probs = model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, test_probs)

    print(f"Test ROC-AUC: {test_roc_auc:.4f}")

    with mlflow.start_run(run_name="candidate_evaluation"):
        mlflow.log_param("evaluated_run_id", run_id)
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.set_tag("evaluation_stage", "candidate")

        if test_roc_auc >= threshold:
            mlflow.set_tag("registration_status", "approved")

            mlflow.log_artifact(input_schema_path.as_posix(), artifact_path="schemas")
            mlflow.log_artifact(output_schema_path.as_posix(), artifact_path="schemas")

            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            model_version = result.version
            
            # ADD @candidate ALIAS INSTEAD OF TRANSITIONING TO STAGE
            client.set_registered_model_alias(
                name=model_name,
                alias="candidate",
                version=model_version
            )
            
            print(f"Model registered as version {model_version} with '@candidate' alias")
        else:
            mlflow.set_tag("registration_status", "rejected")
            print("Model rejected — test ROC-AUC below threshold, not registered.")


if __name__ == "__main__":
    main()