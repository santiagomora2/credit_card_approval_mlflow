import argparse
from mlflow import MlflowClient
from datetime import datetime


def archive_model_version(model_name: str, version: int, reason: str = ""):
    """Archive a specific model version using tags."""
    client = MlflowClient()
    
    try:
        # Check if version exists
        mv = client.get_model_version(model_name, version)
    except Exception as e:
        print(f"Error: Model '{model_name}' version {version} not found")
        return False
    
    # Remove all active aliases (keeps registry clean)
    aliases = mv.aliases
    for alias in aliases:
        client.delete_registered_model_alias(model_name, alias)
        print(f"  Removed alias: {alias}")
    
    # Set archival tags
    client.set_model_version_tag(
        model_name, version, "archived", "true"
    )
    client.set_model_version_tag(
        model_name, version, "archived_at", datetime.now().isoformat()
    )
    
    if reason:
        client.set_model_version_tag(
            model_name, version, "archival_reason", reason
        )
    
    print(f"✓ Model '{model_name}' version {version} archived")
    return True


def unarchive_model_version(model_name: str, version: int, new_alias: str = ""):
    """Unarchive a model version by removing archival tags."""
    client = MlflowClient()
    
    try:
        # Check if version exists
        mv = client.get_model_version(model_name, version)
    except Exception as e:
        print(f"Error: Model '{model_name}' version {version} not found")
        return False
    
    # Remove archival tags
    tags = client.get_model_version_tags(model_name, version)
    
    if "archived" in tags:
        client.delete_model_version_tag(model_name, version, "archived")
    
    if "archived_at" in tags:
        client.delete_model_version_tag(model_name, version, "archived_at")
    
    if "archival_reason" in tags:
        client.delete_model_version_tag(model_name, version, "archival_reason")
    
    # Optionally assign a new alias
    if new_alias:
        client.set_registered_model_alias(
            model_name, new_alias, version
        )
        print(f"  Assigned alias: {new_alias}")
    
    print(f"✓ Model '{model_name}' version {version} unarchived")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Archive or unarchive MLflow model versions using tags")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a model version")
    archive_parser.add_argument("--model-name", required=True)
    archive_parser.add_argument("--version", type=int, required=True)
    archive_parser.add_argument("--reason", default="", help="Reason for archival")
    
    # Unarchive command
    unarchive_parser = subparsers.add_parser("unarchive", help="Unarchive a model version")
    unarchive_parser.add_argument("--model-name", required=True)
    unarchive_parser.add_argument("--version", type=int, required=True)
    unarchive_parser.add_argument("--alias", default="", help="Optional alias to assign after unarchiving")
    
    args = parser.parse_args()
    
    if args.command == "archive":
        archive_model_version(args.model_name, args.version, args.reason)
    
    elif args.command == "unarchive":
        unarchive_model_version(args.model_name, args.version, args.alias)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()