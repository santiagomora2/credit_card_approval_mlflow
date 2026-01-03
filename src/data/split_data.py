import yaml
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("src/config/config.yaml")

    input_path = config["data"]["features_path"]
    target_col = config["data"]["target_column"]

    train_size = config["split"]["train_size"]
    val_size = config["split"]["val_size"]
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Train/val/test sizes must sum to 1.0"

    df = pd.read_csv(input_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train vs temp (val + test)
    sss_1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(val_size + test_size),
        random_state=random_state
    )

    train_idx, temp_idx = next(sss_1.split(X, y))

    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]

    # Second split: val vs test
    relative_test_size = test_size / (val_size + test_size)

    sss_2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_state
    )

    X_temp = df_temp.drop(columns=[target_col])
    y_temp = df_temp[target_col]

    val_idx, test_idx = next(sss_2.split(X_temp, y_temp))

    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]

    df_train.to_csv("data/features/train.csv", index=False)
    df_val.to_csv("data/features/val.csv", index=False)
    df_test.to_csv("data/features/test.csv", index=False)


if __name__ == "__main__":
    main()
