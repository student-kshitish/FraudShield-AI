from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data/processed/transaction_features.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "phase04_figures"

MODEL_FILE = MODEL_DIR / "xgboost_fraud_model.json"
REPORT_FILE = REPORT_DIR / "phase04_model_report.json"
IMPORTANCE_FILE = REPORT_DIR / "phase04_feature_importance.csv"
PREDICTION_FILE = ROOT / "data/processed/test_predictions.csv"

for directory in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_dataset():

    print("\n========== LOADING FEATURE DATA ==========")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = ["event_timestamp", "is_fraud"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["event_timestamp", "is_fraud"]
    ).copy()

    df["is_fraud"] = df["is_fraud"].astype(int)

    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    df = df.sort_values("event_timestamp").reset_index(drop=True)

    excluded = {
        "event_timestamp",
        "sender_account",
        "is_fraud",
        "fraud_ring_id",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric model features found.")

    X = df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # Training data must not contain missing numeric values.
    X = X.fillna(0)

    y = df["is_fraud"].copy()

    print("Dataset shape:", df.shape)
    print("Transactions:", len(df))
    print("Fraud cases:", int(y.sum()))
    print("Legitimate cases:", int((y == 0).sum()))
    print("Features:", len(feature_columns))
    print("Feature names:", feature_columns)

    return df, X, y, feature_columns


# =========================================================
# 2. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# =========================================================

def split_dataset(df, X, y):

    print("\n========== CHRONOLOGICAL DATA SPLIT ==========")

    total = len(df)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:validation_end]
    y_val = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Testing rows:", len(X_test))

    print("\nFraud distribution:")

    print(
        "Train:",
        int(y_train.sum()),
        "/",
        len(y_train),
    )

    print(
        "Validation:",
        int(y_val.sum()),
        "/",
        len(y_val),
    )

    print(
        "Test:",
        int(y_test.sum()),
        "/",
        len(y_test),
    )

    for name, labels in [
        ("training", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        if labels.nunique() < 2:
            raise ValueError(
                f"{name} split does not contain both classes."
            )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# =========================================================
# 3. TRAIN XGBOOST MODEL
# =========================================================

def train_model(X_train, y_train, X_val, y_val):

    print("\n========== TRAINING XGBOOST ==========")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("Negative samples:", negative_count)
    print("Positive samples:", positive_count)
    print("Scale positive weight:", round(scale_pos_weight, 2))

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=3,
        gamma=0.1,

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining started...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\nTraining completed.")

    model.save_model(str(MODEL_FILE))

    print("Model saved:", MODEL_FILE)

    return model


# =========================================================
# 4. SELECT FRAUD THRESHOLD USING VALIDATION DATA
# =========================================================

def select_threshold(model, X_val, y_val):

    print("\n========== THRESHOLD OPTIMIZATION ==========")

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        probabilities,
    )

    # F2 gives more importance to recall than precision.
    beta = 2

    f2_scores = (
        (1 + beta**2) * precision[:-1] * recall[:-1]
        /
        (
            beta**2 * precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f2_scores))

    best_threshold = float(thresholds[best_index])

    print("Default threshold: 0.50")
    print("Selected threshold:", round(best_threshold, 6))
    print("Validation F2:", round(float(f2_scores[best_index]), 4))

    return best_threshold


# =========================================================
# 5. EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test, threshold):

    print("\n========== FINAL TEST EVALUATION ==========")

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_test,
        predictions,
        beta=2,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    print("\nTEST METRICS")
    print("------------------------------")
    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("F2-score:", round(f2, 4))

    print("\nCONFUSION MATRIX")
    print("------------------------------")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "test_transactions": int(len(y_test)),
        "actual_fraud_cases": int(y_test.sum()),
        "predicted_fraud_cases": int(predictions.sum()),
    }

    return metrics, probabilities, predictions, matrix


# =========================================================
# 6. FEATURE IMPORTANCE
# =========================================================

def save_feature_importance(model, feature_columns):

    print("\n========== FEATURE IMPORTANCE ==========")

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print(importance.to_string(index=False))

    plt.figure(figsize=(10, 8))

    top_features = importance.head(15).sort_values(
        "importance"
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.title("XGBoost - Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_importance.png",
        dpi=150,
    )

    plt.close()

    return importance


# =========================================================
# 7. SAVE TEST PREDICTIONS
# =========================================================

def save_predictions(
    df,
    X_test,
    y_test,
    probabilities,
    predictions,
):

    start = len(df) - len(X_test)

    test_rows = df.iloc[start:].copy()

    output = pd.DataFrame({
        "event_timestamp": test_rows["event_timestamp"],
        "actual_fraud": y_test.to_numpy(),
        "fraud_probability": probabilities,
        "predicted_fraud": predictions,
    })

    if "sender_account" in test_rows.columns:
        output["sender_account"] = (
            test_rows["sender_account"].to_numpy()
        )

    output.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    print("\nTest predictions saved:", PREDICTION_FILE)


# =========================================================
# 8. SAVE EVALUATION FIGURES
# =========================================================

def save_evaluation_figures(
    y_test,
    probabilities,
    predictions,
    matrix,
):

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud Detection Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=150,
    )

    plt.close()

    # Confusion matrix
    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.title("Fraud Detection Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=13,
            )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    print("Evaluation figures saved:", FIGURE_DIR)


# =========================================================
# 9. MAIN
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("FRAUDSHIELD AI - PHASE 04")
    print("XGBOOST FRAUD DETECTION MODEL")
    print("=" * 60)

    df, X, y, feature_columns = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(df, X, y)

    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    threshold = select_threshold(
        model,
        X_val,
        y_val,
    )

    metrics, probabilities, predictions, matrix = (
        evaluate_model(
            model,
            X_test,
            y_test,
            threshold,
        )
    )

    importance = save_feature_importance(
        model,
        feature_columns,
    )

    save_predictions(
        df,
        X_test,
        y_test,
        probabilities,
        predictions,
    )

    save_evaluation_figures(
        y_test,
        probabilities,
        predictions,
        matrix,
    )

    report = {
        "phase": 4,
        "status": "MODEL_TRAINING_COMPLETED",
        "model": "XGBoost",
        "input_dataset": str(
            INPUT_FILE.relative_to(ROOT)
        ),
        "total_transactions": int(len(df)),
        "total_fraud_cases": int(y.sum()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split": {
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "strategy": "chronological_70_15_15",
        },
        "class_imbalance": {
            "scale_pos_weight": float(
                (y_train == 0).sum() / (y_train == 1).sum()
            ),
        },
        "evaluation": metrics,
        "top_5_features": (
            importance.head(5)["feature"].tolist()
        ),
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "predictions_file": str(
            PREDICTION_FILE.relative_to(ROOT)
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\n" + "=" * 60)
    print("PHASE 04 SUMMARY")
    print("=" * 60)

    print("Model: XGBoost")
    print("Features:", len(feature_columns))
    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1-score:", round(metrics["f1_score"], 4))
    print("F2-score:", round(metrics["f2_score"], 4))
    print("Selected threshold:", round(threshold, 6))

    print("\nModel:", MODEL_FILE)
    print("Report:", REPORT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()
