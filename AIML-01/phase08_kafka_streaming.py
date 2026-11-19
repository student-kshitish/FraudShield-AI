import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/transaction_features.csv"
MODEL_FILE = ROOT / "models/xgboost_fraud_model.json"
FEATURE_REPORT = ROOT / "reports/phase03_feature_report.json"

OUTPUT_FILE = ROOT / "data/processed/streaming_predictions.csv"
REPORT_FILE = ROOT / "reports/phase08_streaming_report.json"

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraudshield-transactions"
GROUP_ID = "fraudshield-phase08"

DEFAULT_LIMIT = 1000


def load_model_and_features():
    print("\n========== LOADING MODEL AND FEATURES ==========")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    if not FEATURE_REPORT.exists():
        raise FileNotFoundError(FEATURE_REPORT)

    df = pd.read_csv(DATA_FILE)

    with open(FEATURE_REPORT, "r") as f:
        report = json.load(f)

    feature_columns = report.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "feature_columns not found in Phase 03 feature report."
        )

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))

    booster_features = model.get_booster().feature_names

    if booster_features is not None:
        if list(booster_features) != list(feature_columns):
            raise ValueError(
                "Phase 03 feature order does not match XGBoost model."
            )

    missing = [c for c in feature_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing training features: {missing}")

    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if X.shape[1] != 27:
        raise ValueError(
            f"Expected 27 features, found {X.shape[1]}."
        )

    print("Dataset rows:", len(df))
    print("Model:", MODEL_FILE.name)
    print("Features:", len(feature_columns))
    print("Feature order: VERIFIED")

    return df, model, feature_columns


def json_value(value):
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if pd.isna(value):
        return None

    return value


def make_message(row, feature_columns, transaction_number):
    features = {
        column: json_value(row[column])
        for column in feature_columns
    }

    metadata_columns = [
        "event_timestamp",
        "sender_account",
        "receiver_account",
        "amount",
    ]

    metadata = {
        column: json_value(row[column])
        for column in metadata_columns
        if column in row.index
    }

    return {
        "transaction_number": transaction_number,
        "metadata": metadata,
        "features": features,
        "sent_at": time.time(),
    }


def produce_transactions(df, feature_columns, limit):
    print("\n========== KAFKA PRODUCER ==========")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(
            value, allow_nan=False
        ).encode("utf-8"),
        acks="all",
        retries=5,
    )

    count = min(limit, len(df))

    try:
        for index in range(count):
            row = df.iloc[index]

            message = make_message(
                row,
                feature_columns,
                index + 1,
            )

            producer.send(TOPIC, value=message)

            if (index + 1) % 100 == 0:
                print(f"Published: {index + 1}/{count}")

        producer.flush(timeout=60)

    finally:
        producer.close()

    print("Producer completed.")
    print("Messages published:", count)

    return count


def consume_and_predict(model, feature_columns, expected_count):
    print("\n========== KAFKA CONSUMER ==========")

    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
        max_poll_records=100,
        request_timeout_ms=30000,
        consumer_timeout_ms=1000,
    )

    consumer.subscribe([TOPIC])

    # Allow the consumer to join the group before producing.
    for _ in range(10):
        consumer.poll(timeout_ms=1000)
        if consumer.assignment():
            break

    results = []
    processed = 0
    start_time = time.time()

    try:
        while processed < expected_count:
            records = consumer.poll(
                timeout_ms=1000,
                max_records=100,
            )

            for messages in records.values():
                for message in messages:
                    payload = message.value

                    features = payload.get("features", {})
                    metadata = payload.get("metadata", {})

                    missing = [
                        name for name in feature_columns
                        if name not in features
                    ]

                    if missing:
                        raise ValueError(
                            f"Message missing features: {missing}"
                        )

                    feature_row = pd.DataFrame(
                        [[features[name] for name in feature_columns]],
                        columns=feature_columns,
                    ).apply(pd.to_numeric, errors="coerce")

                    probability = float(
                        model.predict_proba(feature_row)[0][1]
                    )

                    prediction = (
                        "FRAUD" if probability >= 0.5
                        else "LEGITIMATE"
                    )

                    risk_level = (
                        "HIGH" if probability >= 0.80
                        else "MEDIUM" if probability >= 0.50
                        else "LOW"
                    )

                    latency_ms = (
                        time.time() - payload["sent_at"]
                    ) * 1000

                    result = {
                        "transaction_number": payload[
                            "transaction_number"
                        ],
                        **metadata,
                        "fraud_probability": probability,
                        "prediction": prediction,
                        "risk_level": risk_level,
                        "processing_latency_ms": latency_ms,
                    }

                    results.append(result)
                    processed += 1

                    if processed % 100 == 0:
                        print(
                            f"Consumed and scored: "
                            f"{processed}/{expected_count}"
                        )

                    if processed >= expected_count:
                        break

            if time.time() - start_time > 180:
                raise TimeoutError(
                    f"Timed out after processing {processed} "
                    f"of {expected_count} messages."
                )

    finally:
        consumer.close()

    return results


def save_outputs(results, elapsed_seconds):
    print("\n========== SAVING STREAMING OUTPUTS ==========")

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    total = len(output_df)
    fraud_count = int(
        (output_df["prediction"] == "FRAUD").sum()
    )
    legitimate_count = total - fraud_count

    avg_latency = float(
        output_df["processing_latency_ms"].mean()
    )

    report = {
        "status": "PHASE08_STREAMING_COMPLETED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transactions_processed": total,
        "fraud_predictions": fraud_count,
        "legitimate_predictions": legitimate_count,
        "average_processing_latency_ms": avg_latency,
        "total_processing_time_seconds": elapsed_seconds,
        "topic": TOPIC,
        "model": "XGBoost",
        "feature_count": 27,
        "feature_source": "Phase 03 engineered transaction dataset",
        "mode": "Kafka feature-enriched transaction replay",
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("Predictions:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)

    print("\n========== PHASE 08 SUMMARY ==========")
    print("Transactions processed:", total)
    print("Fraud predictions:", fraud_count)
    print("Legitimate predictions:", legitimate_count)
    print("Average latency (ms):", round(avg_latency, 3))
    print("Total processing time (seconds):", round(elapsed_seconds, 3))


def main():
    print("=" * 60)
    print("FRAUDSHIELD AI - PHASE 08")
    print("REAL-TIME KAFKA STREAMING")
    print("=" * 60)

    try:
        df, model, feature_columns = load_model_and_features()

        limit = DEFAULT_LIMIT

        if len(sys.argv) > 1:
            limit = int(sys.argv[1])

        if limit <= 0:
            raise ValueError("Transaction limit must be positive.")

        if limit > len(df):
            raise ValueError(
                f"Requested {limit} transactions, "
                f"but dataset has only {len(df)}."
            )

        print("\nWaiting for Kafka producer/consumer pipeline...")

        start_time = time.time()

        # Consumer subscribes before producer publishes.
        # Polling below receives and scores the published messages.
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda value: json.loads(
                value.decode("utf-8")
            ),
            max_poll_records=100,
        )

        consumer.subscribe([TOPIC])

        for _ in range(10):
            consumer.poll(timeout_ms=1000)
            if consumer.assignment():
                break

        count = produce_transactions(
            df,
            feature_columns,
            limit,
        )

        results = []
        processed = 0
        deadline = time.time() + 180

        try:
            while processed < count:
                records = consumer.poll(
                    timeout_ms=1000,
                    max_records=100,
                )

                for messages in records.values():
                    for message in messages:
                        payload = message.value
                        features = payload["features"]
                        metadata = payload["metadata"]

                        feature_row = pd.DataFrame(
                            [[features[name] for name in feature_columns]],
                            columns=feature_columns,
                        ).apply(pd.to_numeric, errors="coerce")

                        probability = float(
                            model.predict_proba(feature_row)[0][1]
                        )

                        results.append({
                            "transaction_number": payload[
                                "transaction_number"
                            ],
                            **metadata,
                            "fraud_probability": probability,
                            "prediction": (
                                "FRAUD" if probability >= 0.5
                                else "LEGITIMATE"
                            ),
                            "risk_level": (
                                "HIGH" if probability >= 0.80
                                else "MEDIUM" if probability >= 0.50
                                else "LOW"
                            ),
                            "processing_latency_ms": (
                                time.time() - payload["sent_at"]
                            ) * 1000,
                        })

                        processed += 1

                        if processed % 100 == 0:
                            print(
                                f"Consumed and scored: "
                                f"{processed}/{count}"
                            )

                        if processed >= count:
                            break

                if time.time() > deadline:
                    raise TimeoutError(
                        f"Processed only {processed}/{count} messages."
                    )

        finally:
            consumer.close()

        elapsed = time.time() - start_time
        save_outputs(results, elapsed)

        print("\nPHASE 08 COMPLETED")
        print("=" * 60)

    except (Exception, KafkaError) as error:
        print("\nPHASE 08 FAILED")
        print(type(error).__name__ + ":", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
