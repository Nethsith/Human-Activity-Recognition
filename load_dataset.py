import pandas as pd
from pathlib import Path


def make_unique(names):
    seen = {}
    unique_names = []

    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")

    return unique_names


def load_har_dataset(data_path):
    data_path = Path(data_path)

    features_df = pd.read_csv(
        data_path / "features.txt",
        sep=r"\s+",
        header=None,
        names=["index", "feature"]
    )
    feature_names = make_unique(features_df["feature"].tolist())

    activity_labels_df = pd.read_csv(
        data_path / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "activity"]
    )
    activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

    X_train = pd.read_csv(
        data_path / "train" / "X_train.txt",
        sep=r"\s+",
        header=None,
        names=feature_names
    )

    y_train = pd.read_csv(
        data_path / "train" / "y_train.txt",
        sep=r"\s+",
        header=None,
        names=["Activity"]
    )

    X_test = pd.read_csv(
        data_path / "test" / "X_test.txt",
        sep=r"\s+",
        header=None,
        names=feature_names
    )

    y_test = pd.read_csv(
        data_path / "test" / "y_test.txt",
        sep=r"\s+",
        header=None,
        names=["Activity"]
    )

    y_train["Activity"] = y_train["Activity"].map(activity_map)
    y_test["Activity"] = y_test["Activity"].map(activity_map)

    return X_train, X_test, y_train["Activity"], y_test["Activity"]