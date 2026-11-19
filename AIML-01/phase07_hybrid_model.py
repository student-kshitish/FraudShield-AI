"""
FRAUDSHIELD AI - PHASE 07
HYBRID FRAUD DETECTION: XGBOOST + GRAPHSAGE

Combines the two independently trained models using a fixed
50:50 probability blend. The test set is used only for evaluation.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]

XGB_PREDICTIONS = ROOT / "data/processed/test_predictions.csv"
GNN_PREDICTIONS = ROOT / "data/processed/gnn_test_predictions.csv"

XGB_REPORT = ROOT / "reports/phase04_model_report.json"
GNN_REPORT = ROOT / "reports/phase06_gnn_report.json"

OUTPUT_MODEL = ROOT / "models/hybrid_fraud_model.json"
OUTPUT_PREDICTIONS = ROOT / "data/processed/hybrid_test_predictions.csv"
OUTPUT_REPORT = ROOT / "reports/phase07_hybrid_report.json"

XGB_WEIGHT = 0.50
GNN_WEIGHT = 0.50
HYBRID_THRESHOLD = 0.50


# =========================================================
# 1. COLUMN DETECTION
# =========================================================

def normalize_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, candidates, description):
    normalized = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Accepted names: {candidates}"
    )


def detect_label_column(df):
    return find_column(
        df,
        [
            "is_fraud",
            "actual_fraud",
            "actual_fraud_label",
            "y_true",
            "true_label",
            "actual_label",
            "actual",
            "label",
            "target",
            "ground_truth",
        ],
        "actual fraud label column",
    )


def detect_probability_column(df):
    candidates = [
        "fraud_probability",
        "y_probability",
        "y_prob",
        "predicted_probability",
        "prediction_probability",
        "probability",
        "probabilities",
        "fraud_score",
        "fraud_prob",
        "is_fraud_probability",
        "prediction_score",
        "score",
        "positive_probability",
        "proba",
    ]

    normalized = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            column = normalized[key]
            values = pd.to_numeric(df[column], errors="coerce")

            if values.notna().all():
                return column

    # Fallback: find a numeric probability-like column.
    excluded_names = {
        "is_fraud",
        "y_true",
        "true_label",
        "actual_label",
        "actual",
        "label",
        "target",
        "ground_truth",
        "y_pred",
        "prediction",
        "predicted_label",
        "event_timestamp",
        "timestamp",
    }

    possible = []

    for column in df.columns:
        if normalize_name(column) in excluded_names:
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        if values.notna().all() and len(values) > 0:
            if values.between(0, 1).all():
                possible.append(column)

    if len(possible) == 1:
        return possible[0]

    raise ValueError(
        "Could not uniquely identify the fraud probability column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Probability-like numeric columns: {possible}"
    )


# =========================================================
# 2. LOAD PREDICTIONS AND REPORTS
# =========================================================

def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r") as file:
        return json.load(file)


def load_predictions(path, model_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{model_name} prediction file is empty.")

    label_column = detect_label_column(df)
    probability_column = detect_probability_column(df)

    labels = pd.to_numeric(
        df[label_column], errors="coerce"
    ).to_numpy(dtype=float)

    probabilities = pd.to_numeric(
        df[probability_column], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(labels).all():
        raise ValueError(f"{model_name}: invalid fraud labels.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{model_name}: labels must be 0 or 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{model_name}: invalid probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"{model_name}: probabilities must be between 0 and 1."
        )

    print(f"\n{model_name} prediction file:")
    print("Rows:", len(df))
    print("Label column:", label_column)
    print("Probability column:", probability_column)
    print("Fraud cases:", int(labels.sum()))

    return {
        "dataframe": df,
        "labels": labels.astype(int),
        "probabilities": probabilities,
        "label_column": label_column,
        "probability_column": probability_column,
    }


# =========================================================
# 3. VERIFY TEST TRANSACTION ALIGNMENT
# =========================================================

def verify_alignment(xgb, gnn):
    """Align GraphSAGE predictions with XGBoost using transaction identifiers."""
    import pandas as pd
    import numpy as np

    keys = ["event_timestamp", "sender_account"]

    # Extract DataFrames from prediction dictionaries.
    xgb_df = xgb["dataframe"].reset_index(drop=True)
    gnn_df = gnn["dataframe"].reset_index(drop=True)

    for column in keys:
        if column not in xgb_df.columns:
            raise ValueError(f"Missing XGBoost transaction identifier: {column}")
        if column not in gnn_df.columns:
            raise ValueError(f"Missing GraphSAGE transaction identifier: {column}")

    xgb_labels = np.asarray(xgb["labels"])
    gnn_labels = np.asarray(gnn["labels"])
    gnn_probabilities = np.asarray(gnn["probabilities"])

    if len(xgb_df) != len(xgb_labels):
        raise ValueError("XGBoost dataframe and labels have different lengths.")

    if len(gnn_df) != len(gnn_labels):
        raise ValueError("GraphSAGE dataframe and labels have different lengths.")

    if len(gnn_df) != len(gnn_probabilities):
        raise ValueError("GraphSAGE dataframe and probabilities have different lengths.")

    def build_keys(df):
        timestamps = pd.to_datetime(
            df["event_timestamp"], utc=True, errors="coerce"
        )

        if timestamps.isna().any():
            raise ValueError("Invalid transaction timestamps detected.")

        accounts = df["sender_account"].astype(str)

        return list(zip(timestamps.astype("int64"), accounts))

    xgb_keys = build_keys(xgb_df)
    gnn_keys = build_keys(gnn_df)

    if len(set(xgb_keys)) != len(xgb_keys):
        raise ValueError("Duplicate transaction identifiers in XGBoost.")

    if len(set(gnn_keys)) != len(gnn_keys):
        raise ValueError("Duplicate transaction identifiers in GraphSAGE.")

    gnn_positions = {key: i for i, key in enumerate(gnn_keys)}

    missing = set(xgb_keys) - set(gnn_keys)
    extra = set(gnn_keys) - set(xgb_keys)

    if missing or extra:
        raise ValueError(
            f"Transaction mismatch: {len(missing)} missing, {len(extra)} extra."
        )

    # Reorder GraphSAGE rows to match XGBoost transaction order.
    positions = np.asarray(
        [gnn_positions[key] for key in xgb_keys],
        dtype=int
    )

    aligned_gnn_df = gnn_df.iloc[positions].reset_index(drop=True)
    aligned_gnn_labels = gnn_labels[positions]
    aligned_gnn_probabilities = gnn_probabilities[positions]

    # Verify fraud labels after alignment.
    if not np.array_equal(xgb_labels, aligned_gnn_labels):
        raise ValueError("Fraud labels do not match after transaction alignment.")

    # Update the original GraphSAGE prediction dictionary.
    gnn["dataframe"] = aligned_gnn_df
    gnn["labels"] = aligned_gnn_labels
    gnn["probabilities"] = aligned_gnn_probabilities

    print("\nTRANSACTION ALIGNMENT VERIFIED")
    print(f"Matched transactions: {len(xgb_keys)}")
    print("Keys: event_timestamp + sender_account")
    print("Fraud labels: MATCH")
    print("GraphSAGE rows, labels and probabilities reordered successfully.")

    return "event_timestamp + sender_account (reordered)"


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f2_score": float(
            fbeta_score(
                y_true, predictions, beta=2, zero_division=0
            )
        ),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_fraud_cases": int(predictions.sum()),
    }


def get_report_threshold(report, default=0.5):
    """Find a threshold stored in an existing phase report."""

    preferred_keys = {
        "threshold",
        "selected_threshold",
        "optimal_threshold",
        "best_threshold",
        "decision_threshold",
    }

    found = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                normalized = normalize_name(key)

                if (
                    normalized in preferred_keys
                    and isinstance(value, (int, float))
                    and 0 <= float(value) <= 1
                ):
                    found.append(float(value))

                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(report)

    return found[0] if found else default


# =========================================================
# 5. HYBRID MODEL
# =========================================================

def combine_probabilities(xgb_probabilities, gnn_probabilities):
    return (
        XGB_WEIGHT * xgb_probabilities
        + GNN_WEIGHT * gnn_probabilities
    )


# =========================================================
# 6. SAVE OUTPUTS
# =========================================================

def save_outputs(
    y_true,
    xgb_probabilities,
    gnn_probabilities,
    hybrid_probabilities,
    alignment_method,
    xgb_metrics,
    gnn_metrics,
    hybrid_metrics,
):
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "model_name": "FraudShield Hybrid Fraud Detector",
        "phase": 7,
        "architecture": "Weighted probability ensemble",
        "components": [
            "XGBoost",
            "GraphSAGE",
        ],
        "weights": {
            "xgboost": XGB_WEIGHT,
            "graphsage": GNN_WEIGHT,
        },
        "decision_threshold": HYBRID_THRESHOLD,
        "combination": (
            "hybrid_probability = "
            "0.5 * xgboost_probability + "
            "0.5 * graphsage_probability"
        ),
        "training_method": (
            "No additional model training. "
            "Combines probabilities from Phase 04 and Phase 06."
        ),
        "threshold_policy": (
            "Fixed 0.5 threshold for initial evaluation; "
            "not optimized on test labels."
        ),
    }

    with open(OUTPUT_MODEL, "w") as file:
        json.dump(model_config, file, indent=2)

    output = pd.DataFrame({
        "is_fraud": y_true,
        "xgboost_probability": xgb_probabilities,
        "graphsage_probability": gnn_probabilities,
        "hybrid_probability": hybrid_probabilities,
        "hybrid_prediction": (
            hybrid_probabilities >= HYBRID_THRESHOLD
        ).astype(int),
    })

    output.to_csv(OUTPUT_PREDICTIONS, index=False)

    report = {
        "phase": 7,
        "status": "HYBRID_EVALUATION_COMPLETED",
        "model": "XGBoost + GraphSAGE",
        "test_transactions": int(len(y_true)),
        "actual_fraud_cases": int(y_true.sum()),
        "alignment_method": alignment_method,
        "architecture": model_config,
        "evaluation": {
            "xgboost": xgb_metrics,
            "graphsage": gnn_metrics,
            "hybrid": hybrid_metrics,
        },
        "output_files": {
            "model_configuration": str(
                OUTPUT_MODEL.relative_to(ROOT)
            ),
            "predictions": str(
                OUTPUT_PREDICTIONS.relative_to(ROOT)
            ),
        },
    }

    with open(OUTPUT_REPORT, "w") as file:
        json.dump(report, file, indent=2)

    return report


# =========================================================
# 7. MAIN
# =========================================================

def main():
    print("=" * 65)
    print("FRAUDSHIELD AI - PHASE 07")
    print("HYBRID FRAUD DETECTION")
    print("=" * 65)

    xgb = load_predictions(XGB_PREDICTIONS, "XGBoost")
    gnn = load_predictions(GNN_PREDICTIONS, "GraphSAGE")

    alignment_method = verify_alignment(xgb, gnn)

    y_true = xgb["labels"]
    xgb_probabilities = xgb["probabilities"]
    gnn_probabilities = gnn["probabilities"]

    hybrid_probabilities = combine_probabilities(
        xgb_probabilities,
        gnn_probabilities,
    )

    xgb_report = load_json(XGB_REPORT)
    gnn_report = load_json(GNN_REPORT)

    xgb_threshold = get_report_threshold(xgb_report)
    gnn_threshold = get_report_threshold(gnn_report)

    print("\n========== MODEL THRESHOLDS ==========")
    print("XGBoost threshold:", xgb_threshold)
    print("GraphSAGE threshold:", gnn_threshold)
    print("Hybrid threshold:", HYBRID_THRESHOLD)

    xgb_metrics = calculate_metrics(
        y_true, xgb_probabilities, xgb_threshold
    )

    gnn_metrics = calculate_metrics(
        y_true, gnn_probabilities, gnn_threshold
    )

    hybrid_metrics = calculate_metrics(
        y_true, hybrid_probabilities, HYBRID_THRESHOLD
    )

    print("\n========== MODEL COMPARISON ==========")

    for name, metrics in [
        ("XGBoost", xgb_metrics),
        ("GraphSAGE", gnn_metrics),
        ("HYBRID", hybrid_metrics),
    ]:
        print(f"\n{name}")
        print("-" * 40)
        print(f"PR-AUC:    {metrics['pr_auc']:.5f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.5f}")
        print(f"Precision: {metrics['precision']:.5f}")
        print(f"Recall:    {metrics['recall']:.5f}")
        print(f"F1:        {metrics['f1_score']:.5f}")
        print(f"F2:        {metrics['f2_score']:.5f}")
        print(
            "Confusion matrix: "
            f"TN={metrics['true_negative']} "
            f"FP={metrics['false_positive']} "
            f"FN={metrics['false_negative']} "
            f"TP={metrics['true_positive']}"
        )

    report = save_outputs(
        y_true,
        xgb_probabilities,
        gnn_probabilities,
        hybrid_probabilities,
        alignment_method,
        xgb_metrics,
        gnn_metrics,
        hybrid_metrics,
    )

    print("\n========== PHASE 07 SAVED OUTPUTS ==========")
    print("Model:", OUTPUT_MODEL)
    print("Predictions:", OUTPUT_PREDICTIONS)
    print("Report:", OUTPUT_REPORT)

    print("\n========== PHASE 07 SUMMARY ==========")
    print("Status:", report["status"])
    print("Transactions:", len(y_true))
    print("Fraud cases:", int(y_true.sum()))
    print("Hybrid PR-AUC:", f"{hybrid_metrics['pr_auc']:.5f}")
    print("Hybrid ROC-AUC:", f"{hybrid_metrics['roc_auc']:.5f}")
    print("Hybrid F1:", f"{hybrid_metrics['f1_score']:.5f}")
    print("Hybrid F2:", f"{hybrid_metrics['f2_score']:.5f}")

    print("\nPHASE 07 COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
