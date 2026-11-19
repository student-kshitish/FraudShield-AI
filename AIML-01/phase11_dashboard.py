import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://127.0.0.1:8000")
DATA_FILE = ROOT / "data" / "processed" / "transaction_features.csv"
DRIFT_FILE = ROOT / "reports" / "phase10_drift_report.json"
FEATURE_REPORT = ROOT / "reports" / "phase03_feature_report.json"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudShield AI")
st.caption("Transaction fraud detection and model monitoring dashboard")

# ---------------------------------------------------------
# API HELPERS
# ---------------------------------------------------------

def get_api_data(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r") as file:
        return json.load(file)


# ---------------------------------------------------------
# TOP-LEVEL STATUS
# ---------------------------------------------------------

health, health_error = get_api_data("/health")
metrics, metrics_error = get_api_data("/metrics")
model_info, model_error = get_api_data("/model-info")

col1, col2, col3 = st.columns(3)

with col1:
    if health and health.get("model_loaded"):
        st.success("API: ONLINE")
    else:
        st.error("API: OFFLINE")
        if health_error:
            st.caption(health_error)

with col2:
    if model_info:
        st.metric("Model", model_info.get("model", "Unknown"))
        st.caption(f"{model_info.get('feature_count', 0)} engineered features")
    else:
        st.metric("Model", "Unavailable")

with col3:
    if metrics:
        st.metric("Predictions served", metrics.get("predictions_completed", 0))
    else:
        st.metric("Predictions served", "Unavailable")

st.divider()

# ---------------------------------------------------------
# PREDICTION MONITORING
# ---------------------------------------------------------

st.header("Prediction Monitoring")

if metrics:
    a, b, c, d = st.columns(4)

    a.metric("Requests received", metrics.get("requests_received", 0))
    b.metric("Fraud predictions", metrics.get("fraud_predictions", 0))
    b.metric(
        "Legitimate predictions",
        metrics.get("legitimate_predictions", 0),
    )
    c.metric(
        "Average model latency",
        f"{metrics.get('average_prediction_latency_ms', 0):.2f} ms",
    )
    d.metric("Model loaded", "Yes" if metrics.get("model_loaded") else "No")
else:
    st.warning("Prediction metrics are unavailable. Check the API service.")

# ---------------------------------------------------------
# DRIFT MONITORING
# ---------------------------------------------------------

st.divider()
st.header("Data Drift Monitoring")

drift = load_json(DRIFT_FILE)

if drift:
    status = drift.get("overall_drift_status", "UNKNOWN")

    if status == "DRIFT_DETECTED":
        st.error(f"Drift status: {status}")
    elif status == "DRIFT_WARNING":
        st.warning(f"Drift status: {status}")
    else:
        st.success(f"Drift status: {status}")

    a, b, c = st.columns(3)
    a.metric("Critical features", drift.get("critical_drift_features", 0))
    b.metric("Warning features", drift.get("warning_features", 0))
    c.metric("Features monitored", drift.get("feature_count", 0))

    st.info(drift.get("retraining_recommendation", "No recommendation available."))

    results = drift.get("feature_results", [])
    drift_df = pd.DataFrame(results)

    if not drift_df.empty:
        display_columns = [
            col for col in [
                "feature",
                "status",
                "psi",
                "ks_statistic",
                "ks_p_value",
                "reference_missing_pct",
                "current_missing_pct",
            ]
            if col in drift_df.columns
        ]

        st.subheader("Feature Drift Details")
        st.dataframe(
            drift_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        chart_df = drift_df.dropna(subset=["psi"]).sort_values(
            "psi", ascending=False
        )

        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="feature",
                y="psi",
                color="status",
                title="Population Stability Index by Feature",
                labels={"psi": "PSI", "feature": "Feature"},
            )
            fig.add_hline(
                y=0.10,
                line_dash="dash",
                annotation_text="Warning threshold",
            )
            fig.add_hline(
                y=0.20,
                line_dash="dash",
                annotation_text="Critical threshold",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Phase 10 drift report not found. Run Phase 10 first.")

# ---------------------------------------------------------
# SAMPLE TRANSACTION PREDICTION
# ---------------------------------------------------------

st.divider()
st.header("Test Transaction Prediction")

st.write(
    "Select a transaction from the processed dataset and send its "
    "engineered features to the FastAPI prediction service."
)

if DATA_FILE.exists() and FEATURE_REPORT.exists():
    try:
        @st.cache_data
        def load_sample_data():
            data = pd.read_csv(DATA_FILE)
            with open(FEATURE_REPORT, "r") as file:
                report = json.load(file)
            return data, report["feature_columns"]

        data, feature_columns = load_sample_data()

        selected_index = st.number_input(
            "Transaction row index",
            min_value=0,
            max_value=len(data) - 1,
            value=0,
            step=1,
        )

        if st.button("Run Fraud Prediction", type="primary"):
            row = data.iloc[int(selected_index)]

            features = {}
            for feature in feature_columns:
                value = row[feature]
                features[feature] = (
                    None if pd.isna(value) else float(value)
                )

            metadata = {}
            for name in [
                "event_timestamp",
                "sender_account",
                "receiver_account",
                "amount",
            ]:
                if name in data.columns:
                    value = row[name]
                    metadata[name] = (
                        None if pd.isna(value) else str(value)
                    )

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "features": features,
                        "metadata": metadata,
                    },
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()["result"]

                if result["prediction"] == "FRAUD":
                    st.error("Prediction: FRAUD")
                else:
                    st.success("Prediction: LEGITIMATE")

                p1, p2, p3 = st.columns(3)
                p1.metric(
                    "Fraud probability",
                    f"{result['fraud_probability']:.2%}",
                )
                p2.metric("Risk level", result["risk_level"])
                p3.metric(
                    "Prediction latency",
                    f"{result['latency_ms']:.2f} ms",
                )

                st.json(result)

            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
            except (KeyError, ValueError) as exc:
                st.error(f"Unexpected API response: {exc}")

    except Exception as exc:
        st.error(f"Could not load transaction dataset: {exc}")
else:
    st.warning(
        "Processed dataset or Phase 03 feature report is missing."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.caption(
    "FraudShield AI | Phase 11 | XGBoost + FastAPI + Streamlit"
)
