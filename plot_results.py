import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# =========================================================
# Create output folder for figures
# =========================================================
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. MODEL COMPARISON BAR CHART
# =========================================================
def plot_model_comparison():
    models = [
        "Logistic Regression\n(Baseline)",
        "SVM\n(Baseline)",
        "Random Forest\n(Baseline)",
        "KNN\n(Baseline)",
        "Logistic Regression\n+ ANOVA",
        "SVM\n+ ANOVA",
        "Logistic Regression\n+ MI",
        "SVM\n+ MI"
    ]

    accuracies = [
        0.954530,
        0.951815,
        0.926026,
        0.883610,
        0.957584,
        0.957923,
        0.954500,
        0.954900
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison on HAR Dataset")
    plt.ylim(0.85, 1.00)
    plt.xticks(rotation=20, ha="right")

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.002,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# 2. BASELINE VS IMPROVED COMPARISON
# =========================================================
def plot_baseline_vs_improved():
    models = ["Logistic Regression", "SVM"]
    baseline = [0.954530, 0.951815]
    improved = [0.957584, 0.957923]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width / 2, baseline, width, label="Baseline")
    bars2 = plt.bar(x + width / 2, improved, width, label="Improved")

    plt.ylabel("Accuracy")
    plt.title("Baseline vs Improved Models")
    plt.xticks(x, models)
    plt.ylim(0.94, 0.965)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0005,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_vs_improved.png"), dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# 3. CONFUSION MATRIX FOR BEST MODEL
#    Best model: SVM + ANOVA
# =========================================================
def plot_best_confusion_matrix():
    cm = np.array([
        [537,   0,   0,   0,   0,   0],
        [  0, 442,  48,   0,   0,   1],
        [  0,  23, 509,   0,   0,   0],
        [  0,   0,   0, 496,   0,   0],
        [  0,   0,   0,   7, 401,  12],
        [  0,   0,   0,  32,   1, 438]
    ])

    labels = [
        "LAYING",
        "SITTING",
        "STANDING",
        "WALKING",
        "WALKING_DOWNSTAIRS",
        "WALKING_UPSTAIRS"
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=30, values_format="d")
    plt.title("Confusion Matrix of Best Model (SVM + ANOVA)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# 4. FEATURE SELECTION IMPACT GRAPH
#    Using your best CV/test accuracy trends from tuning
# =========================================================
def plot_feature_selection_impact():
    feature_counts = [200, 400]

    lr_anova = [0.935188, 0.957584]
    svm_anova = [0.932474, 0.957923]
    lr_mi = [0.930777, 0.954500]
    svm_mi = [0.938921, 0.954900]

    plt.figure(figsize=(8, 5))
    plt.plot(feature_counts, lr_anova, marker="o", label="LR + ANOVA")
    plt.plot(feature_counts, svm_anova, marker="o", label="SVM + ANOVA")
    plt.plot(feature_counts, lr_mi, marker="o", label="LR + MI")
    plt.plot(feature_counts, svm_mi, marker="o", label="SVM + MI")

    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("Accuracy")
    plt.title("Impact of Feature Selection on Model Accuracy")
    plt.xticks(feature_counts)
    plt.ylim(0.92, 0.965)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_selection_impact.png"), dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    plot_model_comparison()
    plot_baseline_vs_improved()
    plot_best_confusion_matrix()
    plot_feature_selection_impact()