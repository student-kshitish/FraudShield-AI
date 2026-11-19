#!/usr/bin/env python3

"""
FraudShield AI
PHASE 02: Synthetic Transaction Generation and EDA

Generates:
- 100,000 synthetic transactions
- Normal and fraudulent transactions
- Coordinated fraud rings
- Transaction analysis
- EDA visualizations
- Data quality report

No Kaggle dataset required.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "phase02_figures"

SEED = 42


def generate_transactions(n=100_000):

    print("\n[1/5] Generating synthetic transactions...")

    rng = np.random.default_rng(SEED)

    fraud_rate = 0.008
    fraud_count = int(n * fraud_rate)

    normal_count = n - fraud_count

    # Account identifiers
    accounts = [
        f"ACC_{i:06d}"
        for i in range(1, 10_001)
    ]

    devices = [
        f"DEV_{i:05d}"
        for i in range(1, 2_001)
    ]

    locations = [
        f"LOC_{i:03d}"
        for i in range(1, 51)
    ]

    # Base transaction data
    sender = rng.choice(accounts, n).astype(object)

    receiver = rng.choice(accounts, n).astype(object)

    # Avoid self-transfers
    same_account = sender == receiver

    while same_account.any():
        receiver[same_account] = rng.choice(
            accounts,
            same_account.sum()
        )

        same_account = sender == receiver

    transaction_types = rng.choice(
        ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"],
        size=n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    amount = rng.lognormal(
        mean=5.0,
        sigma=1.0,
        size=n
    )

    device = rng.choice(devices, n).astype(object)

    location = rng.choice(locations, n).astype(object)

    # Timestamps spread across 90 days
    start = pd.Timestamp("2026-08-01")

    timestamps = (
        start
        + pd.to_timedelta(
            rng.integers(0, 90 * 24 * 60 * 60, n),
            unit="s"
        )
    )

    # Fraud labels
    labels = np.zeros(n, dtype=np.int8)

    fraud_indices = rng.choice(
        n,
        size=fraud_count,
        replace=False
    )

    labels[fraud_indices] = 1

    # Fraudulent transactions generally have larger amounts
    amount[fraud_indices] = rng.lognormal(
        mean=7.0,
        sigma=1.1,
        size=fraud_count
    )

    transaction_types[fraud_indices] = rng.choice(
        ["TRANSFER", "CASH_OUT"],
        size=fraud_count,
        p=[0.55, 0.45]
    )

    # Generate coordinated fraud rings
    fraud_ring_ids = np.full(n, "", dtype=object)

    ring_transaction_count = int(fraud_count * 0.75)

    ring_indices = fraud_indices[:ring_transaction_count]

    individual_indices = fraud_indices[ring_transaction_count:]

    fraud_ring_ids[individual_indices] = "INDIVIDUAL"

    ring_count = 25

    for ring in range(ring_count):

        ring_id = f"RING_{ring + 1:03d}"

        ring_senders = [
            f"FRAUD_ACC_{ring}_{i}"
            for i in range(5)
        ]

        ring_receivers = [
            f"FRAUD_TARGET_{ring}_{i}"
            for i in range(5)
        ]

        ring_device = f"FRAUD_DEVICE_{ring}"

        ring_location = f"FRAUD_LOC_{ring}"

        assigned = ring_indices[
            np.arange(len(ring_indices)) % ring_count == ring
        ]

        sender[assigned] = rng.choice(
            ring_senders,
            len(assigned)
        )

        receiver[assigned] = rng.choice(
            ring_receivers,
            len(assigned)
        )

        device[assigned] = ring_device

        location[assigned] = ring_location

        fraud_ring_ids[assigned] = ring_id

    # Account balances
    sender_before = rng.uniform(
        1_000,
        500_000,
        n
    )

    receiver_before = rng.uniform(
        1_000,
        500_000,
        n
    )

    sender_after = np.maximum(
        sender_before - amount,
        0
    )

    receiver_after = receiver_before + amount

    df = pd.DataFrame({

        "transaction_id": [
            f"TXN_{i:08d}"
            for i in range(1, n + 1)
        ],

        "timestamp": timestamps,

        "sender_id": sender,

        "receiver_id": receiver,

        "amount": np.round(amount, 2),

        "transaction_type": transaction_types,

        "device_id": device,

        "location": location,

        "sender_balance_before": np.round(
            sender_before, 2
        ),

        "sender_balance_after": np.round(
            sender_after, 2
        ),

        "receiver_balance_before": np.round(
            receiver_before, 2
        ),

        "receiver_balance_after": np.round(
            receiver_after, 2
        ),

        "is_fraud": labels,

        "fraud_ring_id": fraud_ring_ids,

    })

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def validate_dataset(df):

    print("\n[2/5] Validating dataset...")

    assert len(df) > 0

    assert df["transaction_id"].is_unique

    assert df["is_fraud"].isin([0, 1]).all()

    assert (df["amount"] > 0).all()

    assert df.isnull().sum().sum() == 0

    assert df["sender_id"].ne(
        df["receiver_id"]
    ).all()

    fraud_rate = df["is_fraud"].mean()

    assert fraud_rate < 0.01

    print("Dataset validation: PASSED")

    print("Total transactions:", len(df))

    print("Fraud transactions:", int(df["is_fraud"].sum()))

    print("Fraud percentage:", round(fraud_rate * 100, 4), "%")


def generate_eda(df):

    print("\n[3/5] Performing exploratory data analysis...")

    fraud = df[df["is_fraud"] == 1]

    normal = df[df["is_fraud"] == 0]

    report = {

        "project": "FraudShield AI",

        "phase": 2,

        "generated_at": datetime.now().isoformat(),

        "dataset": {

            "total_transactions": int(len(df)),

            "fraud_transactions": int(len(fraud)),

            "normal_transactions": int(len(normal)),

            "fraud_percentage": round(
                float(df["is_fraud"].mean() * 100), 4
            ),

            "normal_percentage": round(
                float((1 - df["is_fraud"].mean()) * 100), 4
            ),

            "columns": list(df.columns),

            "missing_values": int(
                df.isnull().sum().sum()
            ),

            "duplicate_transactions": int(
                df["transaction_id"].duplicated().sum()
            ),

            "unique_accounts": int(
                pd.concat([
                    df["sender_id"],
                    df["receiver_id"]
                ]).nunique()
            ),

            "unique_devices": int(
                df["device_id"].nunique()
            ),

            "unique_locations": int(
                df["location"].nunique()
            ),

            "fraud_ring_count": int(
                df.loc[
                    df["fraud_ring_id"].str.startswith(
                        "RING_", na=False
                    ),
                    "fraud_ring_id"
                ].nunique()
            )
        },

        "amount_statistics": {

            "overall_mean": float(df["amount"].mean()),

            "overall_median": float(df["amount"].median()),

            "fraud_mean": float(fraud["amount"].mean()),

            "normal_mean": float(normal["amount"].mean()),

            "maximum": float(df["amount"].max()),

            "minimum": float(df["amount"].min()),

        },

        "transaction_type_distribution": {
            str(k): int(v)
            for k, v in df["transaction_type"].value_counts().items()
        },

        "fraud_by_transaction_type": {
            str(k): int(v)
            for k, v in fraud["transaction_type"].value_counts().items()
        },

        "class_imbalance_ratio": round(
            len(normal) / max(len(fraud), 1), 2
        ),

    }

    return report


def create_visualizations(df):

    print("\n[4/5] Creating EDA visualizations...")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    counts = df["is_fraud"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Normal", "Fraud"],
        counts.values
    )

    plt.title("Transaction Class Distribution")

    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "class_distribution.png")

    plt.close()

    # 2. Amount distribution
    plt.figure(figsize=(9, 5))

    plt.hist(
        df.loc[df["is_fraud"] == 0, "amount"],
        bins=60,
        alpha=0.6,
        label="Normal",
        log=True
    )

    plt.hist(
        df.loc[df["is_fraud"] == 1, "amount"],
        bins=60,
        alpha=0.6,
        label="Fraud",
        log=True
    )

    plt.title("Transaction Amount Distribution")

    plt.xlabel("Transaction Amount")

    plt.ylabel("Frequency (log scale)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "amount_distribution.png")

    plt.close()

    # 3. Fraud by transaction type
    fraud_types = df[df["is_fraud"] == 1][
        "transaction_type"
    ].value_counts()

    plt.figure(figsize=(9, 5))

    fraud_types.plot(kind="bar")

    plt.title("Fraud by Transaction Type")

    plt.xlabel("Transaction Type")

    plt.ylabel("Fraud Count")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "fraud_by_type.png")

    plt.close()

    # 4. Fraud over time
    time_data = df.copy()

    time_data["date"] = time_data["timestamp"].dt.date

    daily = time_data.groupby("date")["is_fraud"].sum()

    plt.figure(figsize=(12, 5))

    daily.plot()

    plt.title("Daily Fraud Transactions")

    plt.xlabel("Date")

    plt.ylabel("Fraud Count")

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "daily_fraud.png")

    plt.close()

    # 5. Fraud ring distribution
    ring_data = df[
        df["fraud_ring_id"].str.startswith(
            "RING_", na=False
        )
    ]

    ring_counts = ring_data["fraud_ring_id"].value_counts()

    plt.figure(figsize=(12, 5))

    ring_counts.sort_index().plot(kind="bar")

    plt.title("Transactions Across Fraud Rings")

    plt.xlabel("Fraud Ring")

    plt.ylabel("Transaction Count")

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "fraud_rings.png")

    plt.close()

    print("EDA visualizations generated:", FIGURE_DIR)


def save_outputs(df, report):

    print("\n[5/5] Saving outputs...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_DIR / "transactions.csv"

    report_path = REPORT_DIR / "phase02_eda_report.json"

    df.to_csv(
        dataset_path,
        index=False
    )

    with open(report_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nDataset saved:", dataset_path)

    print("Report saved:", report_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Number of transactions to generate"
    )

    args = parser.parse_args()

    if args.rows < 10_000:
        parser.error("Generate at least 10,000 transactions.")

    print("=" * 65)

    print("FRAUDSHIELD AI - PHASE 02")

    print("SYNTHETIC DATA GENERATION AND EDA")

    print("=" * 65)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_transactions(args.rows)

    validate_dataset(df)

    report = generate_eda(df)

    create_visualizations(df)

    save_outputs(df, report)

    print("\n" + "=" * 65)

    print("PHASE 02 COMPLETED SUCCESSFULLY")

    print("=" * 65)

    print("Transactions:", len(df))

    print("Fraud:", int(df["is_fraud"].sum()))

    print("Fraud percentage:", round(
        df["is_fraud"].mean() * 100, 4
    ))

    print("Dataset: data/raw/transactions.csv")

    print("Report: reports/phase02_eda_report.json")

    print("Figures: reports/phase02_figures/")


if __name__ == "__main__":
    main()
