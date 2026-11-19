from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import mlflow
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/transaction_features.csv"
MODEL_FILE = ROOT / "models/xgboost_fraud_model.json"
PHASE04_REPORT = ROOT / "reports/phase04_model_report.json"

PREDICTION_FILE = ROOT / "data/processed/test_predictions.csv"
IMPORTANCE_FILE = ROOT / "reports/phase04_feature_importance.csv"

REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "phase05_mlflow_report.json"

MLFLOW_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "FraudShield-Fraud-Detection"

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. VALIDATE PHASE 04 OUTPUTS
# =========================================================

def validate_files():

    print("\n========== VALIDATING PHASE 04 OUTPUTS ==========")

    required_files = [
        DATA_FILE,
        MODEL_FILE,
        PHASE04_REPORT,
        PREDICTION_FILE,
        IMPORTANCE_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required Phase 04 file missing: {file}"
            )

        print("FOUND:", file.relative_to(ROOT))

    report = json.loads(PHASE04_REPORT.read_text())

    if report.get("status") != "MODEL_TRAINING_COMPLETED":
        raise ValueError("Phase 04 report does not show successful training.")

    return report


# =========================================================
# 2. LOAD DATA AND MODEL
# =========================================================

def load_model_and_data():

    print("\n========== LOADING MODEL AND DATA ==========")

    df = pd.read_csv(DATA_FILE)

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

    df = df.sort_values(
        "event_timestamp"
    ).reset_index(drop=True)

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

    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df["is_fraud"]

    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))

    if model.n_features_in_ != len(feature_columns):
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but dataset contains {len(feature_columns)}."
        )

    print("Dataset rows:", len(df))
    print("Features:", len(feature_columns))
    print("Model loaded successfully.")

    return df, X, y, feature_columns, model


# =========================================================
# 3. RECREATE THE CHRONOLOGICAL TEST SPLIT
# =========================================================

def prepare_test_data(df, X, y):

    total = len(df)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    if y_test.nunique() < 2:
        raise ValueError("Test split must contain both classes.")

    print("\nTest transactions:", len(X_test))
    print("Actual fraud cases:", int(y_test.sum()))

    return X_test, y_test


# =========================================================
# 4. CALCULATE METRICS
# =========================================================

def evaluate_model(model, X_test, y_test, phase04_report):

    print("\n========== RECHECKING MODEL PERFORMANCE ==========")

    threshold = float(
        phase04_report["evaluation"]["threshold"]
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "f2_score": float(
            fbeta_score(
                y_test,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
        "threshold": threshold,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_transactions": int(len(y_test)),
        "actual_fraud_cases": int(y_test.sum()),
        "predicted_fraud_cases": int(predictions.sum()),
    }

    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1_score"], 4))
    print("F2:", round(metrics["f2_score"], 4))

    print("\nConfusion matrix:")
    print("TN:", tn, "FP:", fp)
    print("FN:", fn, "TP:", tp)

    return metrics


# =========================================================
# 5. DATASET FINGERPRINT
# =========================================================

def calculate_dataset_hash():

    sha256 = hashlib.sha256()

    with open(DATA_FILE, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# 6. MLFLOW EXPERIMENT TRACKING
# =========================================================

def track_experiment(
    df,
    feature_columns,
    metrics,
    phase04_report,
):

    print("\n========== MLFLOW EXPERIMENT TRACKING ==========")

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset_hash = calculate_dataset_hash()

    with mlflow.start_run(
        run_name="XGBoost-Fraud-Detection-Phase04"
    ) as run:

        # Parameters
        mlflow.log_params({
            "model_type": "XGBoost",
            "feature_count": len(feature_columns),
            "dataset_rows": len(df),
            "training_split": 0.70,
            "validation_split": 0.15,
            "test_split": 0.15,
            "split_strategy": "chronological",
            "threshold": metrics["threshold"],
            "scale_pos_weight": phase04_report[
                "class_imbalance"
            ]["scale_pos_weight"],
            "random_state": 42,
        })

        # Metrics
        mlflow.log_metrics({
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "f2_score": metrics["f2_score"],
            "true_positive": metrics["true_positive"],
            "false_positive": metrics["false_positive"],
            "true_negative": metrics["true_negative"],
            "false_negative": metrics["false_negative"],
        })

        # Tags
        mlflow.set_tags({
            "project": "FraudShield AI",
            "model": "XGBoost",
            "phase": "05",
            "dataset": "Synthetic Financial Transactions",
            "dataset_hash_sha256": dataset_hash,
            "purpose": "Financial Fraud Detection",
        })

        # Save model and supporting artifacts in this run.
        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(PHASE04_REPORT),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(IMPORTANCE_FILE),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(PREDICTION_FILE),
            artifact_path="predictions",
        )

        mlflow.log_dict(
            metrics,
            "evaluation/metrics.json",
        )

        mlflow.log_dict(
            {
                "feature_columns": feature_columns,
                "dataset_hash_sha256": dataset_hash,
                "dataset_rows": len(df),
            },
            "metadata/dataset_metadata.json",
        )

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    print("\nMLflow run completed.")
    print("Experiment ID:", experiment_id)
    print("Run ID:", run_id)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": TRACKING_URI,
        "dataset_hash_sha256": dataset_hash,
    }


# =========================================================
# 7. SAVE PHASE REPORT
# =========================================================

def save_report(metrics, tracking, feature_columns):

    report = {
        "phase": 5,
        "status": "MLFLOW_TRACKING_COMPLETED",
        "model": "XGBoost",
        "feature_count": len(feature_columns),
        "evaluation_metrics": metrics,
        "mlflow": tracking,
        "artifacts_logged": [
            "XGBoost model",
            "Phase 04 model report",
            "Feature importance",
            "Test predictions",
            "Evaluation metrics",
            "Dataset metadata",
        ],
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\nPhase 05 report saved:", REPORT_FILE)


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("FRAUDSHIELD AI - PHASE 05")
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    phase04_report = validate_files()

    df, X, y, feature_columns, model = (
        load_model_and_data()
    )

    X_test, y_test = prepare_test_data(
        df,
        X,
        y,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        phase04_report,
    )

    tracking = track_experiment(
        df,
        feature_columns,
        metrics,
        phase04_report,
    )

    save_report(
        metrics,
        tracking,
        feature_columns,
    )

    print("\n" + "=" * 60)
    print("PHASE 05 SUMMARY")
    print("=" * 60)

    print("MLflow status: SUCCESS")
    print("Experiment:", tracking["experiment_name"])
    print("Run ID:", tracking["run_id"])
    print("Features:", len(feature_columns))
    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Report:", REPORT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()
