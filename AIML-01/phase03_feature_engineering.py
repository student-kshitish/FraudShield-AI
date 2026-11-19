from pathlib import Path
from datetime import timedelta
import json
import sys
import traceback

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data/raw/transactions.csv"
FEATURE_FILE = ROOT / "data/processed/transaction_features.csv"
PARQUET_FILE = ROOT / "feature_store/transaction_features.parquet"
REPORT_FILE = ROOT / "reports/phase03_feature_report.json"
FEAST_CONFIG = ROOT / "feature_store/feature_store.yaml"

FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
PARQUET_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def find_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"Required column not found. Expected one of: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    print("\n========== RAW DATA ==========")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    timestamp_col = find_column(
        df,
        ["timestamp", "transaction_timestamp", "event_timestamp"]
    )

    sender_col = find_column(
        df,
        ["sender_account", "sender", "sender_id"]
    )

    receiver_col = find_column(
        df,
        ["receiver_account", "receiver", "receiver_id"]
    )

    amount_col = find_column(df, ["amount", "transaction_amount"])

    label_col = find_column(
        df,
        ["is_fraud", "fraud", "target"]
    )

    device_col = find_column(
        df,
        ["device_id", "device"],
        required=False
    )

    location_col = find_column(
        df,
        ["location", "transaction_location"],
        required=False
    )

    type_col = find_column(
        df,
        ["transaction_type", "type"],
        required=False
    )

    # Standardize column names for the feature pipeline.
    rename = {
        timestamp_col: "event_timestamp",
        sender_col: "sender_account",
        receiver_col: "receiver_account",
        amount_col: "amount",
        label_col: "is_fraud",
    }

    if device_col:
        rename[device_col] = "device_id"

    if location_col:
        rename[location_col] = "location"

    if type_col:
        rename[type_col] = "transaction_type"

    df = df.rename(columns=rename)

    for column in ["device_id", "location", "transaction_type"]:
        if column not in df.columns:
            df[column] = "UNKNOWN"

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"], errors="coerce", utc=True
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")

    df = df.dropna(
        subset=[
            "event_timestamp",
            "sender_account",
            "receiver_account",
            "amount",
            "is_fraud",
        ]
    ).copy()

    df["is_fraud"] = df["is_fraud"].astype(int)

    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("Fraud label must contain only 0 and 1.")

    if (df["amount"] <= 0).any():
        raise ValueError("Non-positive transaction amounts found.")

    for column in [
        "sender_account",
        "receiver_account",
        "device_id",
        "location",
        "transaction_type",
    ]:
        df[column] = df[column].fillna("UNKNOWN").astype(str)

    df = df.sort_values("event_timestamp").reset_index(drop=True)

    print("Validated shape:", df.shape)
    print("Fraud cases:", int(df["is_fraud"].sum()))

    return df


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------

def add_rolling_feature(df, group_column, window, operation):
    """
    Calculate rolling features using only previous transactions.

    Uses original row positions to avoid duplicate-index problems.
    closed='left' excludes the current transaction.
    """

    result = np.zeros(len(df), dtype=float)

    grouped = df.groupby(group_column, sort=False).indices

    for group_name, positions in grouped.items():

        # Retrieve this group's transactions in chronological order.
        group_df = df.iloc[positions]

        # Calculate rolling values using timestamps.
        rolling_values = (
            group_df
            .set_index("event_timestamp")["amount"]
            .rolling(
                window=window,
                closed="left"
            )
            .agg(operation)
        )

        # Map calculated values back to original row positions.
        result[positions] = (
            rolling_values
            .fillna(0)
            .to_numpy(dtype=float)
        )

    return pd.Series(result, index=df.index).fillna(0)

def engineer_features(df):
    print("\n========== FEATURE ENGINEERING ==========")

    df = df.copy()

    # Time features
    df["transaction_hour"] = df["event_timestamp"].dt.hour
    df["transaction_day"] = df["event_timestamp"].dt.day
    df["transaction_day_of_week"] = (
        df["event_timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 22)
        | (df["transaction_hour"] <= 5)
    ).astype(int)

    # Amount features
    df["amount_log"] = np.log1p(df["amount"])

    # Sender historical features
    df["sender_previous_transaction_count"] = (
        df.groupby("sender_account").cumcount()
    )

    df["sender_txn_count_1h"] = add_rolling_feature(
        df, "sender_account", "1h", "count"
    )

    df["sender_amount_sum_1h"] = add_rolling_feature(
        df, "sender_account", "1h", "sum"
    )

    df["sender_avg_amount_24h"] = add_rolling_feature(
        df, "sender_account", "24h", "mean"
    )

    df["sender_txn_count_24h"] = add_rolling_feature(
        df, "sender_account", "24h", "count"
    )

    # Compare current amount with historical sender average.
    df["amount_vs_sender_avg_24h"] = (
        df["amount"] / (df["sender_avg_amount_24h"] + 1.0)
    )

    # Receiver historical activity
    df["receiver_txn_count_1h"] = add_rolling_feature(
        df, "receiver_account", "1h", "count"
    )

    df["receiver_txn_count_24h"] = add_rolling_feature(
        df, "receiver_account", "24h", "count"
    )

    # Device activity
    df["device_txn_count_1h"] = add_rolling_feature(
        df, "device_id", "1h", "count"
    )

    # Location activity
    df["location_txn_count_1h"] = add_rolling_feature(
        df, "location", "1h", "count"
    )

    # Repeated sender-receiver relationship
    df["sender_receiver_pair_previous_count"] = (
        df.groupby(
            ["sender_account", "receiver_account"]
        ).cumcount()
    )

    # Transaction-type encoding
    type_features = pd.get_dummies(
        df["transaction_type"],
        prefix="transaction_type",
        dtype=int
    )

    df = pd.concat([df, type_features], axis=1)

    # Identify numeric features without target leakage.
    excluded_columns = {
        "is_fraud",
        "fraud_ring_id",
        "event_timestamp",
        "transaction_id",
        "sender_account",
        "receiver_account",
        "device_id",
        "location",
        "transaction_type",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric features were generated.")

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Keep target separate from the model features.
    output_columns = [
        "event_timestamp",
        "sender_account",
        "receiver_account",
        "is_fraud",
    ] + feature_columns

    # Remove duplicate column names while preserving order.
    output_columns = list(dict.fromkeys(output_columns))

    features = df[output_columns].copy()

    print("Total generated numeric features:", len(feature_columns))
    print("Feature columns:", feature_columns)

    return df, features, feature_columns


# ---------------------------------------------------------
# 3. SAVE TRAINING FEATURES AND REPORT
# ---------------------------------------------------------

def save_outputs(df, features, feature_columns):
    features.to_csv(FEATURE_FILE, index=False)

    # Feast uses Parquet as its local file source.
    feast_data = df[
        ["event_timestamp", "sender_account"] + feature_columns
    ].copy()

    # Feast feature values are stored as float64 for this prototype.
    for column in feature_columns:
        feast_data[column] = pd.to_numeric(
            feast_data[column], errors="coerce"
        ).fillna(0).astype("float64")

    feast_data.to_parquet(PARQUET_FILE, index=False)

    report = {
        "phase": 3,
        "status": "FEATURE_ENGINEERING_COMPLETED",
        "input_rows": int(len(df)),
        "fraud_rows": int(df["is_fraud"].sum()),
        "fraud_percentage": round(
            float(df["is_fraud"].mean() * 100), 4
        ),
        "numeric_feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "training_feature_file": str(FEATURE_FILE.relative_to(ROOT)),
        "feast_parquet_file": str(PARQUET_FILE.relative_to(ROOT)),
        "target_column": "is_fraud",
        "target_excluded_from_feast_features": True,
        "leakage_checks": {
            "is_fraud_excluded": "is_fraud" not in feature_columns,
            "fraud_ring_id_excluded": "fraud_ring_id" not in feature_columns,
            "rolling_features_use_prior_events": True,
        },
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\nSaved training features:", FEATURE_FILE)
    print("Saved Feast Parquet:", PARQUET_FILE)
    print("Saved report:", REPORT_FILE)

    return report


# ---------------------------------------------------------
# 4. REGISTER AND TEST FEAST
# ---------------------------------------------------------

def setup_feast(feature_columns, df):
    print("\n========== FEAST FEATURE STORE ==========")

    try:
        from datetime import timedelta as td

        from feast import (
            Entity,
            FeatureStore,
            FeatureView,
            Field,
            FileSource,
        )
        from feast.types import Float64

        sender_entity = Entity(
            name="sender",
            join_keys=["sender_account"],
            description="Transaction sender account",
        )

        source = FileSource(
            name="fraud_transaction_features_source",
            path=str(PARQUET_FILE.resolve()),
            timestamp_field="event_timestamp",
        )

        feature_view = FeatureView(
            name="fraud_transaction_features",
            entities=[sender_entity],
            ttl=td(days=365),
            schema=[
                Field(name=column, dtype=Float64)
                for column in feature_columns
            ],
            source=source,
            online=True,
            description="Historical transaction risk features",
        )

        store = FeatureStore(
            repo_path=str(FEAST_CONFIG.parent.resolve())
        )

        # Register the entity and feature view.
        store.apply([sender_entity, feature_view])
        print("Feast entity and feature view registered.")

        # Materialize historical feature data into the online store.
        start_date = df["event_timestamp"].min().to_pydatetime()
        end_date = (
            df["event_timestamp"].max() + td(seconds=1)
        ).to_pydatetime()

        store.materialize(
            start_date=start_date,
            end_date=end_date,
        )

        print("Feast materialization completed.")

        # Read one sender's features from the online store.
        sample_sender = str(df.iloc[-1]["sender_account"])

        selected_features = [
            f"fraud_transaction_features:{column}"
            for column in feature_columns[:5]
        ]

        online_result = store.get_online_features(
            features=selected_features,
            entity_rows=[
                {"sender_account": sample_sender}
            ],
        ).to_dict()

        print("Online feature retrieval successful.")
        print("Sample online result:", online_result)

        return {
            "feast_status": "SUCCESS",
            "entity": "sender",
            "feature_view": "fraud_transaction_features",
            "online_store": "SQLite",
            "materialization": "SUCCESS",
            "online_retrieval": "SUCCESS",
        }

    except Exception as error:
        print("\nFEAST SETUP FAILED:")
        print(str(error))
        traceback.print_exc()

        return {
            "feast_status": "FAILED",
            "error": str(error),
        }


# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("FRAUDSHIELD AI - PHASE 03")
    print("FEATURE ENGINEERING + FEAST")
    print("=" * 60)

    df = load_data()

    engineered_df, features, feature_columns = engineer_features(df)

    report = save_outputs(
        engineered_df,
        features,
        feature_columns,
    )

    feast_result = setup_feast(
        feature_columns,
        engineered_df,
    )

    report.update(feast_result)

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n" + "=" * 60)
    print("PHASE 03 SUMMARY")
    print("=" * 60)
    print("Transactions processed:", len(df))
    print("Fraud transactions:", int(df["is_fraud"].sum()))
    print("Numeric features:", len(feature_columns))
    print("Feast status:", feast_result["feast_status"])
    print("Training features:", FEATURE_FILE)
    print("Feature report:", REPORT_FILE)
    print("=" * 60)

    if feast_result["feast_status"] != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
