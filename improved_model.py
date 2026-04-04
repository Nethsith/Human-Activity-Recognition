
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from load_dataset import load_har_dataset


# --------------------------------------------------
# Evaluate improved model
# --------------------------------------------------
def evaluate_tuned_model(
    model_name,
    selector_name,
    selector_score_func,
    base_model,
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
    label_names
):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=selector_score_func)),
        ("model", base_model)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=label_names,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"Improved Model: {model_name} + {selector_name}")
    print("=" * 70)
    print("Best Params:", grid.best_params_)
    print(f"Best CV Score: {grid.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Confusion Matrix:")
    print(cm)

    result = {
        "Model": model_name,
        "Feature_Selection": selector_name,
        "Best_Params": str(grid.best_params_),
        "Best_CV_Score": grid.best_score_,
        "Test_Accuracy": acc,
        "Macro_Precision": report["macro avg"]["precision"],
        "Macro_Recall": report["macro avg"]["recall"],
        "Macro_F1": report["macro avg"]["f1-score"],
        "Weighted_F1": report["weighted avg"]["f1-score"]
    }

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "UCI HAR Dataset"

    # Load data
    X_train, X_test, y_train_text, y_test_text = load_har_dataset(DATA_PATH)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)
    label_names = list(label_encoder.classes_)

    print("Dataset loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Classes      :", label_names)

    results = []

    # --------------------------------------------------
    # Logistic Regression + ANOVA
    # --------------------------------------------------
    lr_anova_params = {
        "selector__k": [50, 100, 200, 300, 400],
        "model__C": [0.1, 1, 10]
    }

    results.append(
        evaluate_tuned_model(
            model_name="Logistic Regression",
            selector_name="ANOVA",
            selector_score_func=f_classif,
            base_model=LogisticRegression(max_iter=1000, random_state=42),
            param_grid=lr_anova_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names
        )
    )

    # --------------------------------------------------
    # Logistic Regression + Mutual Information
    # --------------------------------------------------
    lr_mi_params = {
        "selector__k": [50, 100, 200, 300, 400],
        "model__C": [0.1, 1, 10]
    }

    results.append(
        evaluate_tuned_model(
            model_name="Logistic Regression",
            selector_name="Mutual Information",
            selector_score_func=mutual_info_classif,
            base_model=LogisticRegression(max_iter=1000, random_state=42),
            param_grid=lr_mi_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names
        )
    )

    # --------------------------------------------------
    # SVM + ANOVA
    # --------------------------------------------------
    svm_anova_params = {
        "selector__k": [50, 100, 200, 300, 400],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"]
    }

    results.append(
        evaluate_tuned_model(
            model_name="SVM",
            selector_name="ANOVA",
            selector_score_func=f_classif,
            base_model=SVC(random_state=42),
            param_grid=svm_anova_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names
        )
    )

    # --------------------------------------------------
    # SVM + Mutual Information
    # --------------------------------------------------
    svm_mi_params = {
        "selector__k": [50, 100, 200, 300, 400],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"]
    }

    results.append(
        evaluate_tuned_model(
            model_name="SVM",
            selector_name="Mutual Information",
            selector_score_func=mutual_info_classif,
            base_model=SVC(random_state=42),
            param_grid=svm_mi_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Test_Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("IMPROVED MODELS SUMMARY")
    print("=" * 70)
    print(results_df[[
        "Model", "Feature_Selection", "Best_CV_Score",
        "Test_Accuracy", "Macro_F1", "Weighted_F1"
    ]])

    results_df.to_csv("output/improved_models_summary.csv", index=False)
