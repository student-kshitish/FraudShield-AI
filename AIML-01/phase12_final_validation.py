import json
import socket
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

results = []

def check(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    results.append({
        "check": name,
        "status": status,
        "details": str(details)
    })
    print(f"[{status}] {name}" + (f" - {details}" if details else ""))


def check_file(relative_path):
    path = ROOT / relative_path
    check(f"File: {relative_path}", path.is_file(),
          f"Size: {path.stat().st_size:,} bytes" if path.is_file()
          else "File not found")


def check_json(relative_path):
    path = ROOT / relative_path
    try:
        with path.open() as f:
            json.load(f)
        check(f"Valid JSON: {relative_path}", True)
    except Exception as exc:
        check(f"Valid JSON: {relative_path}", False, exc)


def check_http(name, url):
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "FraudShield-Phase12-Validation"}
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read().decode("utf-8", errors="replace")
            check(name, response.status == 200,
                  f"HTTP {response.status}")
            return body
    except Exception as exc:
        check(name, False, exc)
        return None


print("=" * 65)
print("FRAUDSHIELD AI - PHASE 12 FINAL VALIDATION")
print("=" * 65)

# 1. Phase source files
print("\n1. PHASE SOURCE FILES")
for phase in range(1, 12):
    check_file(f"AIML-01/phase{phase:02d}_" + {
        1: "setup.py",
        2: "dataset_eda.py",
        3: "feature_engineering.py",
        4: "train_model.py",
        5: "mlflow_tracking.py",
        6: "gnn_model.py",
        7: "hybrid_model.py",
        8: "kafka_streaming.py",
        9: "fastapi_server.py",
        10: "drift_monitoring.py",
        11: "dashboard.py",
    }[phase])

# 2. Model artifacts
print("\n2. MODEL ARTIFACTS")
for path in [
    "models/xgboost_fraud_model.json",
    "models/graphsage_fraud_model.pt",
    "models/hybrid_fraud_model.json",
]:
    check_file(path)

# 3. Reports
print("\n3. PHASE REPORTS")
for path in [
    "reports/phase03_feature_report.json",
    "reports/phase07_hybrid_report.json",
    "reports/phase10_drift_report.json",
]:
    check_json(path)

# 4. Processed data and prediction outputs
print("\n4. PROCESSED DATA")
for path in [
    "data/processed/transaction_features.csv",
    "data/processed/test_predictions.csv",
    "data/processed/gnn_test_predictions.csv",
    "data/processed/hybrid_test_predictions.csv",
    "data/processed/phase10_feature_drift.csv",
]:
    check_file(path)

# 5. Docker deployment files
print("\n5. DOCKER CONFIGURATION")
for path in [
    "Dockerfile.phase11",
    "docker-compose.phase11.yml",
    "requirements-phase11.txt",
]:
    check_file(path)

# 6. API endpoint checks
print("\n6. FASTAPI")
check_http("FastAPI health endpoint", "http://localhost:8000/health")
check_http("FastAPI model-info endpoint", "http://localhost:8000/model-info")
check_http("FastAPI metrics endpoint", "http://localhost:8000/metrics")

# 7. Dashboard health
print("\n7. STREAMLIT DASHBOARD")
check_http("Streamlit health endpoint",
           "http://localhost:8501/_stcore/health")

# 8. Kafka connectivity
print("\n8. KAFKA")
try:
    with socket.create_connection(("localhost", 9092), timeout=5):
        check("Kafka port 9092", True, "TCP connection successful")
except Exception as exc:
    check("Kafka port 9092", False, exc)

# 9. Consolidated result
failures = sum(item["status"] == "FAIL" for item in results)
passed = sum(item["status"] == "PASS" for item in results)

overall = "PASS" if failures == 0 else "FAIL"

report = {
    "project": "FraudShield AI",
    "phase": 12,
    "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "overall_status": overall,
    "checks_passed": passed,
    "checks_failed": failures,
    "total_checks": len(results),
    "results": results,
    "notes": [
        "Validation checks file presence, JSON readability, HTTP endpoints and Kafka TCP connectivity.",
        "A successful health check does not prove model accuracy or production readiness.",
        "Model performance should be assessed using the recorded evaluation reports.",
        "Kafka streaming is a replay prototype using engineered transaction features."
    ]
}

output = REPORT_DIR / "phase12_validation_report.json"
with output.open("w") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 65)
print("PHASE 12 VALIDATION SUMMARY")
print("=" * 65)
print(f"Checks passed : {passed}")
print(f"Checks failed : {failures}")
print(f"Overall       : {overall}")
print(f"Report        : {output}")
print("=" * 65)

if failures:
    raise SystemExit(1)
