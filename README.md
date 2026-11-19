# FraudShield AI
### AI-Powered Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GraphSAGE-red)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-brightgreen)
![Deployment](https://img.shields.io/badge/Deployment-Docker-blue)
![Status](https://img.shields.io/badge/Project-Completed-success)

FraudShield AI is an end-to-end machine learning project designed to detect potentially fraudulent financial transactions using machine learning, graph-based learning, feature engineering, streaming infrastructure, model serving, and drift monitoring.

The project combines traditional machine learning and graph neural networks to identify suspicious transaction patterns and demonstrate an end-to-end fraud detection workflow.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Key Features](#key-features)
4. [Dataset Information](#dataset-information)
5. [System Architecture](#system-architecture)
6. [Technology Stack](#technology-stack)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Evaluation](#model-evaluation)
9. [Project Phases](#project-phases)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Running the Application](#running-the-application)
13. [Docker Deployment](#docker-deployment)
14. [API Documentation](#api-documentation)
15. [Kafka Streaming](#kafka-streaming)
16. [Feature Store](#feature-store)
17. [Model Drift Monitoring](#model-drift-monitoring)
18. [Final Validation](#final-validation)
19. [Limitations](#limitations)
20. [Future Improvements](#future-improvements)
21. [Author](#author)
22. [License](#license)

---

## Project Overview

Financial fraud is a major challenge for banks, payment providers, and financial institutions.

Traditional fraud detection systems often depend on fixed rules and manually defined thresholds. These systems may struggle to identify complex transaction patterns and relationships between accounts.

FraudShield AI explores a machine learning-based approach to fraud detection.

The system includes:

- Transaction data processing.
- Feature engineering.
- Machine learning-based fraud classification.
- Graph-based fraud detection.
- Hybrid model experimentation.
- Feature store integration.
- MLflow experiment tracking.
- Kafka-based transaction replay.
- FastAPI model serving.
- Streamlit monitoring dashboard.
- Data drift monitoring.
- Docker-based deployment.
- Automated project validation.

The project uses synthetic transaction data and is intended as an educational and technical demonstration.

---

## Objectives

The main objectives are:

1. Develop a complete fraud detection workflow.
2. Process and prepare financial transaction data.
3. Engineer useful transaction features.
4. Train machine learning models to identify fraudulent transactions.
5. Explore graph neural networks for relationship-based fraud detection.
6. Experiment with combining model predictions.
7. Integrate a feature store.
8. Track experiments and model performance.
9. Demonstrate transaction streaming with Kafka.
10. Expose prediction functionality through a REST API.
11. Build a monitoring dashboard.
12. Monitor changes in incoming data distributions.
13. Validate the complete project workflow.

---

## Key Features

| Feature | Description |
|---|---|
| Transaction Processing | Processes synthetic financial transaction data. |
| Feature Engineering | Creates 27 numerical features for model training and prediction. |
| XGBoost | Uses gradient-boosted trees for fraud classification. |
| GraphSAGE | Explores graph-based relationships between transactions and entities. |
| Hybrid Model | Experiments with combining model predictions. |
| Feature Store | Integrates Feast for feature management. |
| Experiment Tracking | Uses MLflow for experiment tracking. |
| Kafka | Demonstrates transaction streaming through replay. |
| FastAPI | Provides REST endpoints for model inference. |
| Streamlit | Provides a monitoring and prediction dashboard. |
| Drift Monitoring | Uses PSI and KS testing to identify distribution changes. |
| Docker | Containerizes the API and dashboard. |
| Validation | Runs automated checks across project components. |

---

## Dataset Information

FraudShield AI uses a synthetic financial transaction dataset.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total Transactions | 100,000 |
| Fraudulent Transactions | 800 |
| Legitimate Transactions | 99,200 |
| Fraud Percentage | 0.8% |
| Engineered Numerical Features | 27 |

The dataset is imbalanced, reflecting a common challenge in fraud detection: fraudulent transactions represent a small proportion of total transactions.

### Data Processing

The project includes data processing and feature engineering to prepare transaction records for machine learning.

The engineered features are used by the model training and prediction workflows.

The model-serving API expects the required engineered numerical features rather than automatically constructing every history-dependent feature from raw transaction identifiers.

---

## System Architecture

The project follows an end-to-end workflow:

```text
                 Synthetic Transaction Data
                            |
                            v
                  Data Processing
                            |
                            v
                  Feature Engineering
                            |
                            v
                    Feature Store
                       (Feast)
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          XGBoost         GraphSAGE     Hybrid Model
             |              |              |
             +--------------+--------------+
                            |
                            v
                  Model Evaluation
                            |
                            v
                  MLflow Tracking
                            |
                            v
                   Kafka Streaming
                            |
                            v
                     FastAPI
                            |
                            v
                  Streamlit Dashboard
                            |
                            v
                   Drift Monitoring
```

### Main Components

**Data Layer**
- Synthetic transaction data.
- Processed datasets.
- Engineered transaction features.

**Machine Learning Layer**
- XGBoost fraud classifier.
- GraphSAGE graph neural network.
- Hybrid prediction experiment.

**Infrastructure Layer**
- Feast feature store.
- MLflow experiment tracking.
- Kafka streaming prototype.

**Application Layer**
- FastAPI REST API.
- Streamlit dashboard.
- Docker deployment.

**Monitoring Layer**
- Prediction metrics.
- Data drift detection.
- PSI analysis.
- KS statistical testing.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Machine Learning | XGBoost, Scikit-learn |
| Deep Learning | PyTorch |
| Graph Learning | GraphSAGE |
| Data Processing | Pandas, NumPy |
| Feature Store | Feast |
| Experiment Tracking | MLflow |
| Streaming | Apache Kafka |
| API Framework | FastAPI |
| Dashboard | Streamlit |
| API Server | Uvicorn |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Validation | Python scripts |

---

## Machine Learning Models

### 1. XGBoost

XGBoost is used as a gradient-boosted decision tree model for transaction classification.

It learns patterns from engineered transaction features and produces fraud predictions.

Model artifact:

```text
models/xgboost_fraud_model.json
```

### 2. GraphSAGE

GraphSAGE is a graph neural network approach used to explore relationships between entities and transaction patterns.

Graph-based learning can help identify suspicious patterns that may not be obvious from individual transaction attributes alone.

Model artifact:

```text
models/graphsage_fraud_model.pt
```

### 3. Hybrid Model

The hybrid model represents an experiment combining predictions from multiple approaches.

Its performance is evaluated separately rather than assuming that combining models always improves results.

Model artifact:

```text
models/hybrid_fraud_model.json
```

---

## Model Evaluation

The project evaluates models using metrics appropriate for an imbalanced classification problem.

These include:

- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- PR-AUC

### Hybrid Model Results

Recorded Phase 07 evaluation:

| Metric | Result |
|---|---:|
| Test Transactions | 15,000 |
| Fraudulent Test Transactions | 117 |
| PR-AUC | 0.75729 |
| ROC-AUC | 0.98012 |
| F1-score | 0.66917 |
| F2-score | 0.72123 |

### GraphSAGE Results

Previously recorded GraphSAGE evaluation:

| Metric | Result |
|---|---:|
| PR-AUC | 0.84298 |
| ROC-AUC | 0.98052 |
| Precision | 0.94845 |
| Recall | 0.78632 |
| F1-score | 0.85981 |
| F2-score | 0.81416 |

### Evaluation Interpretation

The recorded GraphSAGE results outperform the recorded hybrid results on the reported metrics.

This demonstrates why individual model evaluation is important when experimenting with hybrid approaches.

The metrics are results from this synthetic dataset and evaluation setup. They should not be interpreted as guaranteed performance on real financial transactions.

---

## Project Phases

The project was developed in 12 phases.

### Phase 01 — Project Setup
- Initial project structure.
- Development environment.
- Dependency setup.

### Phase 02 — Data Preparation
- Synthetic transaction data.
- Data processing.
- Preparation for model development.

### Phase 03 — Feature Engineering
- Transaction feature generation.
- Numerical feature preparation.
- Feature report generation.
- 27 engineered numerical features.

Report:

```text
reports/phase03_feature_report.json
```

### Phase 04 — Machine Learning
- XGBoost model development.
- Model training.
- Fraud classification.

### Phase 05 — Graph-Based Learning
- Graph-based data preparation.
- GraphSAGE model development.
- Graph-based fraud detection.

### Phase 06 — Feature Store and Experiment Tracking
- Feast feature store integration.
- MLflow experiment tracking.
- Model experiment management.

### Phase 07 — Hybrid Model and Evaluation
- Hybrid model experimentation.
- Prediction generation.
- Model evaluation.
- Performance comparison.

### Phase 08 — Kafka Streaming
- Kafka infrastructure.
- Transaction replay.
- Fraud and legitimate transaction processing.
- Processing statistics.

This phase demonstrates replay of pre-engineered transaction features. It is not a complete raw-transaction real-time feature-generation pipeline.

### Phase 09 — FastAPI Model Serving
- REST API development.
- Single transaction prediction.
- Batch prediction.
- Health and model information endpoints.
- Metrics endpoint.

### Phase 10 — Model Drift Monitoring
- Reference and current data comparison.
- Population Stability Index.
- Kolmogorov-Smirnov testing.
- Drift reporting.

### Phase 11 — Dashboard and Docker Deployment
- Streamlit dashboard.
- API integration.
- Model information display.
- Drift visualization.
- Docker containerization.
- Docker Compose deployment.

### Phase 12 — Final Validation
- Project structure checks.
- Model artifact checks.
- Report checks.
- API health checks.
- Dashboard checks.
- Kafka connectivity checks.

Final validation result:

```text
Checks passed : 30
Checks failed : 0
Overall       : PASS
```

Report:

```text
reports/phase12_validation_report.json
```

---

## Project Structure

The main project structure is:

```text
FraudShield-AI/
│
├── AIML-01/
│   └── phase12_final_validation.py
│
├── data/
│   ├── processed/
│   │   ├── test_predictions.csv
│   │   ├── gnn_test_predictions.csv
│   │   └── hybrid_test_predictions.csv
│   └── ...
│
├── feature_store/
│   └── ...
│
├── models/
│   ├── xgboost_fraud_model.json
│   ├── graphsage_fraud_model.pt
│   └── hybrid_fraud_model.json
│
├── reports/
│   ├── phase03_feature_report.json
│   ├── phase12_validation_report.json
│   └── ...
│
├── Dockerfile.phase11
├── docker-compose.phase11.yml
├── docker-compose.yml
├── requirements-phase11.txt
├── README.md
└── .gitignore
```

Additional phase scripts, configuration files, datasets, and supporting components are maintained in their respective project directories.

---

## Installation

### Requirements

- Ubuntu or another compatible Linux environment.
- Python 3.12.
- Git.
- Docker.
- Docker Compose.

### 1. Clone the Repository

```bash
git clone git@github.com:student-kshitish/FraudShield-AI.git
cd FraudShield-AI
```

Alternatively, use HTTPS:

```bash
git clone https://github.com/student-kshitish/FraudShield-AI.git
cd FraudShield-AI
```

### 2. Create a Virtual Environment

```bash
python3.12 -m venv .venv
```

### 3. Activate the Environment

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Application Dependencies

```bash
pip install -r requirements-phase11.txt
```

Note: `requirements-phase11.txt` supports the Phase 11 API and dashboard application. Earlier training, feature-store, graph-learning, and streaming phases may require their respective dependencies and configurations.

---

## Running the Application

### Start the FastAPI Application

Run the API using the project's configured application module:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The exact import module may depend on the API file location in your checkout.

### Open API Documentation

```text
http://localhost:8000/docs
```

### Run the Streamlit Dashboard

Run the dashboard using its project entry-point script:

```bash
streamlit run dashboard.py --server.port 8501
```

The exact dashboard script path may depend on the project directory structure.

Dashboard:

```text
http://localhost:8501
```

---

## Docker Deployment

The project includes Docker configuration for deploying the API and dashboard.

### Build and Start the Application

```bash
docker compose -f docker-compose.phase11.yml up --build -d
```

### Check Running Containers

```bash
docker ps
```

### Check Container Health

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### View Logs

```bash
docker compose -f docker-compose.phase11.yml logs -f
```

### Stop the Application

```bash
docker compose -f docker-compose.phase11.yml down
```

### Application Addresses

| Service | Address |
|---|---|
| FastAPI | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

The Kafka service is configured separately through the root Docker Compose configuration.

---

## API Documentation

The FastAPI application provides REST endpoints for model serving and application monitoring.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Root endpoint |
| `/health` | GET | API health check |
| `/model-info` | GET | Model information |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch transaction prediction |
| `/metrics` | GET | Application metrics |

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Information

```bash
curl http://localhost:8000/model-info
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Interactive API Documentation

```text
http://localhost:8000/docs
```

### Prediction Input

The prediction endpoints require the engineered numerical features expected by the model.

The project uses 27 engineered numerical features.

Use the API documentation to inspect the exact request schema and submit a valid prediction request.

Important: The current API expects engineered features. It does not automatically generate all historical and relationship-based features from only raw transaction amounts, account IDs, or other identifiers.

---

## Kafka Streaming

The project includes a Kafka-based transaction streaming prototype.

The streaming workflow demonstrates:

1. Transaction replay.
2. Kafka message processing.
3. Fraud and legitimate transaction classification.
4. Processing statistics.
5. Latency measurement.

### Recorded Streaming Results

| Metric | Result |
|---|---:|
| Transactions Processed | 1,000 |
| Fraudulent Transactions | 20 |
| Legitimate Transactions | 980 |
| Average Latency | 2,622.58 ms |
| Total Processing Time | 5.803 seconds |

These are recorded results from the project's streaming replay experiment.

The implementation uses pre-engineered features and should not be described as a fully implemented raw-transaction real-time feature engineering system.

---

## Feature Store

The project integrates Feast as its feature store component.

The feature store supports the management and retrieval of model features.

Feature-store integration is part of the project's broader feature engineering and model-serving workflow.

---

## Model Drift Monitoring

FraudShield AI includes a data drift monitoring workflow.

The monitoring process compares reference data with current data to identify changes in feature distributions.

### Recorded Monitoring Setup

| Metric | Value |
|---|---:|
| Reference Records | 70,000 |
| Current Records | 30,000 |
| Features Monitored | 27 |
| Critical PSI Features | 2 |
| Warning PSI Features | 1 |
| Overall Status | DRIFT_DETECTED |

### Population Stability Index (PSI)

The project uses the following PSI interpretation:

| PSI Value | Interpretation |
|---|---|
| PSI < 0.10 | Stable |
| PSI >= 0.10 | Warning |
| PSI >= 0.20 | Critical |

### Kolmogorov-Smirnov Test

The KS test is also used to identify statistically significant distribution differences.

A p-value below 0.05 is treated as a warning.

### Important Note

Detected data drift indicates that feature distributions have changed. It does not automatically prove that model performance has degraded.

Automatic model retraining is not implemented as part of the current project.

---

## Final Validation

The final project validation script is:

```text
AIML-01/phase12_final_validation.py
```

Run it from the project root:

```bash
python AIML-01/phase12_final_validation.py
```

### Final Validation Results

```text
Checks passed : 30
Checks failed : 0
Overall       : PASS
```

Generated report:

```text
reports/phase12_validation_report.json
```

The validation checks cover project components such as:

- Required project files.
- Model artifacts.
- Reports.
- Data artifacts.
- Docker configuration.
- API health.
- Dashboard availability.
- Kafka connectivity.

A passing validation result confirms that the tested project checks passed in the recorded environment. It does not establish production readiness or guarantee performance on real-world financial data.

---

## Limitations

The current implementation has several limitations:

1. The dataset is synthetic rather than real banking transaction data.
2. Model performance may differ significantly on real financial datasets.
3. The dataset contains class imbalance and may not represent every real-world fraud pattern.
4. The API expects engineered features rather than generating all historical features from raw transactions.
5. Kafka streaming demonstrates replay of pre-engineered features rather than a complete real-time feature-generation pipeline.
6. Drift detection identifies changes in feature distributions but does not prove model degradation.
7. Automatic model retraining is not implemented.
8. The system has not been independently validated for production banking environments.
9. Security, compliance, scalability, and operational requirements would need additional work before real financial deployment.

---

## Future Improvements

Potential future improvements include:

- Integration with real-world or approved anonymized transaction datasets.
- Real-time feature engineering from raw transaction events.
- More advanced graph-based fraud detection.
- Improved model calibration.
- Explainable AI for fraud predictions.
- Advanced threshold optimization.
- Automated model retraining pipelines.
- Model performance monitoring.
- Alerting and notification integration.
- Authentication and authorization for API endpoints.
- More extensive automated testing.
- Production-grade logging and observability.
- Cloud deployment and scalable infrastructure.
- Security and compliance reviews.

---

## Author

**Kshitish Behera**

GitHub: [@student-kshitish](https://github.com/student-kshitish)

Project Repository: [FraudShield AI](https://github.com/student-kshitish/FraudShield-AI)

---

## License

No open-source license has currently been specified for this repository.

Unless a license is added, all rights remain with the copyright holder. Please contact the author before redistributing or reusing the project beyond applicable legal exceptions.

---

## Disclaimer

FraudShield AI is an educational and demonstration project.

It is not a certified banking fraud detection product and should not be used to make real financial decisions without appropriate validation, security reviews, compliance checks, and production engineering.

**Built to explore AI-powered financial fraud detection.**
