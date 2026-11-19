import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "transaction_features.csv"
FEATURE_REPORT = ROOT / "reports" / "phase03_feature_report.json"

REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "data" / "processed"

JSON_REPORT = REPORT_DIR / "phase10_drift_report.json"
CSV_REPORT = OUTPUT_DIR / "phase10_feature_drift.csv"

REFERENCE_RATIO = 0.70

PSI_WARNING = 0.10
PSI_CRITICAL = 0.20
KS_SIGNIFICANCE = 0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("FraudShieldDriftMonitor")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if not FEATURE_REPORT.exists():
        raise FileNotFoundError(f"Feature report not found: {FEATURE_REPORT}")

    df = pd.read_csv(DATA_FILE)

    with open(FEATURE_REPORT, "r") as file:
        report = json.load(file)

    feature_columns = report.get("feature_columns", [])

    if len(feature_columns) != 27:
        raise ValueError(
            f"Expected 27 engineered features, found {len(feature_columns)}"
        )

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, feature_columns


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

def split_reference_current(df):
    timestamp_column = None

    for candidate in ["event_timestamp", "timestamp", "transaction_timestamp"]:
        if candidate in df.columns:
            timestamp_column = candidate
            break

    if timestamp_column:
        parsed = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().sum() >= len(df) * 0.8:
            df = df.copy()
            df["_parsed_timestamp"] = parsed
            df = df.sort_values("_parsed_timestamp").reset_index(drop=True)
            split_method = f"Chronological split using {timestamp_column}"
        else:
            split_method = "Original row order; timestamp parsing insufficient"
    else:
        split_method = "Original row order; no timestamp column found"

    split_index = int(len(df) * REFERENCE_RATIO)

    reference = df.iloc[:split_index].copy()
    current = df.iloc[split_index:].copy()

    if len(reference) < 30 or len(current) < 30:
        raise ValueError(
            "Insufficient data for drift analysis. "
            "Both reference and current datasets need at least 30 rows."
        )

    return reference, current, split_method


# ---------------------------------------------------------
# POPULATION STABILITY INDEX
# ---------------------------------------------------------

def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return None

    if np.all(reference == reference[0]):
        if np.all(current == reference[0]):
            return 0.0

        return 1.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    epsilon = 0.0001

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ---------------------------------------------------------
# FEATURE DRIFT ANALYSIS
# ---------------------------------------------------------

def analyze_feature(reference, current, feature):
    ref_values = reference[feature].dropna().to_numpy(dtype=float)
    cur_values = current[feature].dropna().to_numpy(dtype=float)

    if len(ref_values) < 10 or len(cur_values) < 10:
        return {
            "feature": feature,
            "status": "INSUFFICIENT_DATA",
            "psi": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "reference_missing_pct": round(
                float(reference[feature].isna().mean() * 100), 3
            ),
            "current_missing_pct": round(
                float(current[feature].isna().mean() * 100), 3
            ),
        }

    psi = calculate_psi(ref_values, cur_values)
    ks_result = ks_2samp(ref_values, cur_values)

    ks_statistic = float(ks_result.statistic)
    ks_p_value = float(ks_result.pvalue)

    if psi is not None and psi >= PSI_CRITICAL:
        status = "CRITICAL_DRIFT"
    elif psi is not None and (
        psi >= PSI_WARNING or ks_p_value < KS_SIGNIFICANCE
    ):
        status = "WARNING"
    else:
        status = "STABLE"

    return {
        "feature": feature,
        "status": status,
        "psi": round(psi, 6) if psi is not None else None,
        "ks_statistic": round(ks_statistic, 6),
        "ks_p_value": round(ks_p_value, 8),
        "reference_missing_pct": round(
            float(reference[feature].isna().mean() * 100), 3
        ),
        "current_missing_pct": round(
            float(current[feature].isna().mean() * 100), 3
        ),
        "reference_mean": round(float(np.mean(ref_values)), 6),
        "current_mean": round(float(np.mean(cur_values)), 6),
        "reference_std": round(float(np.std(ref_values)), 6),
        "current_std": round(float(np.std(cur_values)), 6),
    }


# ---------------------------------------------------------
# OVERALL DRIFT STATUS
# ---------------------------------------------------------

def determine_overall_status(results):
    critical = sum(
        item["status"] == "CRITICAL_DRIFT" for item in results
    )

    warnings_count = sum(
        item["status"] == "WARNING" for item in results
    )

    insufficient = sum(
        item["status"] == "INSUFFICIENT_DATA" for item in results
    )

    if critical > 0:
        return "DRIFT_DETECTED", critical, warnings_count, insufficient

    if warnings_count > 0:
        return "DRIFT_WARNING", critical, warnings_count, insufficient

    return "STABLE", critical, warnings_count, insufficient


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 10 drift monitoring.")

    df, feature_columns = load_data()

    reference, current, split_method = split_reference_current(df)

    logger.info("Total transactions: %s", len(df))
    logger.info("Reference transactions: %s", len(reference))
    logger.info("Current transactions: %s", len(current))

    results = []

    for feature in feature_columns:
        result = analyze_feature(reference, current, feature)
        results.append(result)

        logger.info(
            "%s | PSI: %s | Status: %s",
            feature,
            result["psi"],
            result["status"],
        )

    overall_status, critical, warning_count, insufficient = (
        determine_overall_status(results)
    )

    if critical > 0:
        retraining_recommendation = (
            "RETRAINING_RECOMMENDED: Critical feature drift detected. "
            "Validate data quality and model performance before retraining."
        )
    elif warning_count > 0:
        retraining_recommendation = (
            "INVESTIGATION_RECOMMENDED: Feature drift warnings detected. "
            "Investigate the affected features and evaluate model performance."
        )
    else:
        retraining_recommendation = (
            "NO_IMMEDIATE_ACTION: No significant drift detected by the "
            "configured PSI and KS checks."
        )

    report = {
        "status": "PHASE10_DRIFT_MONITORING_COMPLETED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "total_transactions": int(len(df)),
        "reference_transactions": int(len(reference)),
        "current_transactions": int(len(current)),
        "reference_ratio": REFERENCE_RATIO,
        "split_method": split_method,
        "feature_count": len(feature_columns),
        "overall_drift_status": overall_status,
        "critical_drift_features": critical,
        "warning_features": warning_count,
        "insufficient_data_features": insufficient,
        "psi_thresholds": {
            "stable": f"PSI < {PSI_WARNING}",
            "warning": f"PSI >= {PSI_WARNING}",
            "critical": f"PSI >= {PSI_CRITICAL}",
        },
        "ks_significance_level": KS_SIGNIFICANCE,
        "retraining_recommendation": retraining_recommendation,
        "automatic_retraining_executed": False,
        "feature_results": results,
    }

    with open(JSON_REPORT, "w") as file:
        json.dump(report, file, indent=4)

    pd.DataFrame(results).to_csv(CSV_REPORT, index=False)

    print("\n========== PHASE 10 SUMMARY ==========")
    print(f"Transactions analyzed: {len(df)}")
    print(f"Reference transactions: {len(reference)}")
    print(f"Current transactions: {len(current)}")
    print(f"Features monitored: {len(feature_columns)}")
    print(f"Critical drift features: {critical}")
    print(f"Warning features: {warning_count}")
    print(f"Overall status: {overall_status}")
    print(f"Recommendation: {retraining_recommendation}")
    print(f"JSON report: {JSON_REPORT}")
    print(f"CSV report: {CSV_REPORT}")
    print("\nPHASE 10 COMPLETED")
    print("=======================================")


if __name__ == "__main__":
    main()
