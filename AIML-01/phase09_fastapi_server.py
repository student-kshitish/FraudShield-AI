import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "xgboost_fraud_model.json"
FEATURE_REPORT = ROOT / "reports" / "phase03_feature_report.json"

THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudShieldAPI")

app = FastAPI(
    title="FraudShield AI API",
    description="REST API for transaction fraud prediction using XGBoost.",
    version="1.0.0",
)

model = None
feature_columns = []
request_count = 0
prediction_count = 0
fraud_count = 0
total_prediction_time_ms = 0.0


# ---------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------

class TransactionRequest(BaseModel):
    features: dict[str, Optional[float]] = Field(
        ...,
        description="All 27 engineered numerical features used during training.",
    )
    metadata: dict = Field(default_factory=dict)


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=500
    )


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_resources():
    global model, feature_columns

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not FEATURE_REPORT.exists():
        raise FileNotFoundError(f"Feature report not found: {FEATURE_REPORT}")

    with open(FEATURE_REPORT, "r") as file:
        report = json.load(file)

    feature_columns = report.get("feature_columns", [])

    if len(feature_columns) != 27:
        raise ValueError(
            f"Expected 27 features in Phase 03 report, found {len(feature_columns)}"
        )

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))

    booster_features = model.get_booster().feature_names

    if booster_features is not None:
        if list(booster_features) != list(feature_columns):
            raise ValueError(
                "Model feature names/order do not match Phase 03 feature report."
            )

    logger.info("XGBoost model loaded successfully.")
    logger.info("Feature count: %s", len(feature_columns))


@app.on_event("startup")
def startup_event():
    load_resources()


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def prepare_features(feature_data: dict) -> pd.DataFrame:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    supplied = set(feature_data.keys())
    required = set(feature_columns)

    missing = sorted(required - supplied)
    extra = sorted(supplied - required)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature names do not match the trained model.",
                "missing_features": missing,
                "unexpected_features": extra,
                "required_feature_count": len(feature_columns),
            },
        )

    row = {}

    for name in feature_columns:
        value = feature_data[name]

        if value is None:
            row[name] = np.nan
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be numeric or null.",
            )

        if not math.isfinite(numeric_value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be finite or null.",
            )

        row[name] = numeric_value

    return pd.DataFrame([row], columns=feature_columns)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def score_transaction(transaction: TransactionRequest):
    global prediction_count, fraud_count, total_prediction_time_ms

    input_df = prepare_features(transaction.features)

    start = time.perf_counter()
    probability = float(model.predict_proba(input_df)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000

    prediction = "FRAUD" if probability >= THRESHOLD else "LEGITIMATE"

    if probability >= 0.8:
        risk_level = "HIGH"
    elif probability >= THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    prediction_count += 1
    total_prediction_time_ms += latency_ms

    if prediction == "FRAUD":
        fraud_count += 1

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "risk_level": risk_level,
        "threshold": THRESHOLD,
        "latency_ms": round(latency_ms, 3),
        "metadata": transaction.metadata,
    }


# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "FraudShield AI",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return {
        "model": "XGBoost",
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "fraud_threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(transaction: TransactionRequest):
    global request_count
    request_count += 1

    try:
        result = score_transaction(transaction)
        return {
            "status": "success",
            "result": result,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        )


@app.post("/predict/batch")
def predict_batch(batch: BatchPredictionRequest):
    global request_count
    request_count += 1

    results = []

    try:
        for index, transaction in enumerate(batch.transactions):
            result = score_transaction(transaction)
            results.append({
                "transaction_index": index,
                **result,
            })

        return {
            "status": "success",
            "transactions_processed": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal server error.",
        )


@app.get("/metrics")
def metrics():
    average_latency = (
        total_prediction_time_ms / prediction_count
        if prediction_count else 0.0
    )

    return {
        "requests_received": request_count,
        "predictions_completed": prediction_count,
        "fraud_predictions": fraud_count,
        "legitimate_predictions": prediction_count - fraud_count,
        "average_prediction_latency_ms": round(average_latency, 3),
        "model_loaded": model is not None,
    }


# ---------------------------------------------------------
# LOCAL EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "phase09_fastapi_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
