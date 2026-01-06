from mlflow import MlflowClient
import argparse

def manage_model_alias(model_name: str, version: int, alias: str, action: str = "add"):
    """Add or remove an alias for a specific model version."""
    client = MlflowClient()
    
    if action == "add":
        # Add or update the alias
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version
        )
        print(f"Model '{model_name}' version {version} assigned alias '{alias}'")
    elif action == "remove":
        # Remove the alias if it exists
        client.delete_registered_model_alias(
            name=model_name,
            alias=alias
        )
        print(f"Removed alias '{alias}' from model '{model_name}'")
    else:
        raise ValueError(f"Unknown action: {action}. Use 'add' or 'remove'.")

def main():
    parser = argparse.ArgumentParser(
        description="Manage MLflow model version aliases"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Registered model name (e.g. credit_card_approval_model)"
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version number to manage alias for"
    )
    parser.add_argument(
        "--alias",
        type=str,
        required=True,
        help="Alias to manage: champion, candidate, canary, production, staging..."
    )
    parser.add_argument(
        "--action",
        type=str,
        required=False,
        default="add",
        choices=["add", "remove"],
        help="Action to perform: add or remove alias (default: add)"
    )
    
    args = parser.parse_args()
    
    manage_model_alias(args.model_name, args.version, args.alias, args.action)

if __name__ == "__main__":
    main()