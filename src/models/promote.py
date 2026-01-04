from mlflow.tracking import MlflowClient
import argparse

def main(src_model: str, src_alias: str, dst_model: str):
    client = MlflowClient()

    src_model_uri = f"models:/{src_model}@{src_alias}"

    client.copy_model_version(
        src_model_uri=src_model_uri,
        dst_name=dst_model
    )

    print(
        f"Model promoted:\n"
        f"  Source: {src_model}@{src_alias}\n"
        f"  Destination: {dst_model}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote MLflow model via copy")

    parser.add_argument(
        "--src-name",
        type=str,
        required=True,
        help="Source model name (e.g. credit_card_approval_model)"
    )
    parser.add_argument(
        "--src-alias",
        type=str,
        required=True,
        help="Source alias (e.g. candidate)"
    )
    parser.add_argument(
        "--dst-name",
        type=str,
        required=True,
        help="Destination model name (e.g. credit_card_approval_model_production)"
    )

    args = parser.parse_args()

    main(args.src_name, args.src_alias, args.dst_name)
