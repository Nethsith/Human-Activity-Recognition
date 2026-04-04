import pandas as pd
from pathlib import Path


def make_unique(names):
    """
    Make duplicate column names unique by adding suffixes.
    Example: feature, feature -> feature, feature_1
    """
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
    """
    Load the UCI HAR Dataset from the given folder path.

    Returns:
        X_train, X_test, y_train, y_test,
        train_df, test_df,
        features_df, activity_labels_df
    """
    data_path = Path(data_path)

    # -----------------------------
    # 1. Load feature names
    # -----------------------------
    features_df = pd.read_csv(
        data_path / "features.txt",
        sep=r"\s+",
        header=None,
        names=["index", "feature"]
    )

    feature_names = make_unique(features_df["feature"].tolist())

    # -----------------------------
    # 2. Load activity labels
    # -----------------------------
    activity_labels_df = pd.read_csv(
        data_path / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "activity"]
    )

    activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

    # -----------------------------
    # 3. Load training data
    # -----------------------------
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

    # -----------------------------
    # 4. Load test data
    # -----------------------------
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

    # -----------------------------
    # 5. Map numeric activity labels to text
    # -----------------------------
    y_train["Activity"] = y_train["Activity"].map(activity_map)
    y_test["Activity"] = y_test["Activity"].map(activity_map)

    # -----------------------------
    # 6. Load subject IDs (optional but useful)
    # -----------------------------
    subject_train = pd.read_csv(
        data_path / "train" / "subject_train.txt",
        sep=r"\s+",
        header=None,
        names=["Subject"]
    )

    subject_test = pd.read_csv(
        data_path / "test" / "subject_test.txt",
        sep=r"\s+",
        header=None,
        names=["Subject"]
    )

    # -----------------------------
    # 7. Combine into full train/test dataframes
    # -----------------------------
    train_df = pd.concat([subject_train, X_train, y_train], axis=1)
    test_df = pd.concat([subject_test, X_test, y_test], axis=1)

    return (
        X_train, X_test, y_train, y_test,
        train_df, test_df,
        features_df, activity_labels_df
    )


def save_dataframes(train_df, test_df, output_folder="output"):
    """
    Save cleaned train and test dataframes as CSV files.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train_clean.csv"
    test_file = output_path / "test_clean.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Saved: {train_file}")
    print(f"Saved: {test_file}")


if __name__ == "__main__":
    # Change this path if needed
    DATA_PATH = "UCI HAR Dataset"

    try:
        (
            X_train, X_test, y_train, y_test,
            train_df, test_df,
            features_df, activity_labels_df
        ) = load_har_dataset(DATA_PATH)

        print("Dataset loaded successfully.\n")

        print("X_train shape:", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape :", y_test.shape)

        print("\nTrain dataframe shape:", train_df.shape)
        print("Test dataframe shape :", test_df.shape)

        print("\nFirst 5 rows of train data:")
        print(train_df.head())

        print("\nTraining activity distribution:")
        print(train_df["Activity"].value_counts())

        # Save cleaned CSV files
        save_dataframes(train_df, test_df)

    except FileNotFoundError as e:
        print("File not found. Check your dataset path.")
        print(e)

    except Exception as e:
        print("An error occurred:")
        print(e)