import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from load_dataset import load_har_dataset



def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Model": name,
        "Accuracy": acc
    }


if __name__ == "__main__":
    DATA_PATH = "UCI HAR Dataset"

    # 1. Load data
    X_train, X_test, y_train_text, y_test_text = load_har_dataset(DATA_PATH)

    # 2. Encode labels to numbers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    # 3. Define baseline models
    baseline_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # 4. Train and evaluate
    results = []

    for name, model in baseline_models.items():
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    # 5. Save summary table
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(f"\n{'='*60}")
    print("Baseline Summary")
    print(f"{'='*60}")
    print(results_df)

    results_df.to_csv("output/baseline_results_summary.csv", index=False)
