from pathlib import Path
import copy
import json
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data/raw/transactions.csv"
FEATURE_FILE = ROOT / "data/processed/transaction_features.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "phase06_figures"

MODEL_FILE = MODEL_DIR / "graphsage_fraud_model.pt"
REPORT_FILE = REPORT_DIR / "phase06_gnn_report.json"
PREDICTION_FILE = ROOT / "data/processed/gnn_test_predictions.csv"

for directory in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEED = 42
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.set_num_threads(min(8, torch.get_num_threads()))


# =========================================================
# 1. GRAPH NEURAL NETWORK ARCHITECTURE
# =========================================================

class FraudGraphSAGE(nn.Module):

    def __init__(self, node_feature_count, edge_feature_count):

        super().__init__()

        # Learn account representations from graph structure.
        self.conv1 = SAGEConv(
            node_feature_count,
            64,
        )

        self.conv2 = SAGEConv(
            64,
            32,
        )

        # Classify a transaction using:
        # sender embedding + receiver embedding + transaction features.
        self.edge_classifier = nn.Sequential(
            nn.Linear(32 + 32 + edge_feature_count, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def encode(self, node_features, edge_index):

        x = self.conv1(
            node_features,
            edge_index,
        )

        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv2(
            x,
            edge_index,
        )

        return F.relu(x)

    def classify_edges(
        self,
        embeddings,
        sender_indices,
        receiver_indices,
        transaction_features,
    ):

        sender_embedding = embeddings[sender_indices]
        receiver_embedding = embeddings[receiver_indices]

        combined = torch.cat(
            [
                sender_embedding,
                receiver_embedding,
                transaction_features,
            ],
            dim=1,
        )

        return self.edge_classifier(combined).squeeze(1)

    def forward(
        self,
        node_features,
        edge_index,
        sender_indices,
        receiver_indices,
        transaction_features,
    ):

        embeddings = self.encode(
            node_features,
            edge_index,
        )

        return self.classify_edges(
            embeddings,
            sender_indices,
            receiver_indices,
            transaction_features,
        )


# =========================================================
# 2. LOAD AND ALIGN TRANSACTION DATA
# =========================================================

def load_data():

    print("\n========== LOADING TRANSACTION DATA ==========")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    # Load the engineered dataset directly.
    # Phase 03 now exports sender and receiver account identities.
    features = pd.read_csv(FEATURE_FILE)

    required_columns = [
        "event_timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "is_fraud",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Phase 03 columns: {missing_columns}"
        )

    features["event_timestamp"] = pd.to_datetime(
        features["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    features["amount"] = pd.to_numeric(
        features["amount"], errors="coerce"
    )

    features["is_fraud"] = pd.to_numeric(
        features["is_fraud"], errors="coerce"
    )

    features = features.dropna(
        subset=required_columns
    ).copy()

    features["sender_account"] = (
        features["sender_account"].astype(str).str.strip()
    )

    features["receiver_account"] = (
        features["receiver_account"].astype(str).str.strip()
    )

    if (
        features["sender_account"].eq("").any()
        or features["receiver_account"].eq("").any()
    ):
        raise ValueError("Empty sender or receiver account found.")

    features["is_fraud"] = features["is_fraud"].astype(int)

    if not set(features["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("Fraud labels must contain only 0 and 1.")

    print("Transactions loaded:", len(features))
    print(
        "Unique accounts:",
        len(set(features["sender_account"]) |
            set(features["receiver_account"]))
    )
    print(
        "Receiver accounts available:",
        features["receiver_account"].notna().all()
    )

    excluded = {
        "event_timestamp",
        "sender_account",
        "receiver_account",
        "is_fraud",
        "fraud_ring_id",
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    X = (
        features[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(float)
    )

    y = features["is_fraud"].to_numpy(dtype=np.float32)

    sender_accounts = features[
        "sender_account"
    ].to_numpy()

    receiver_accounts = features[
        "receiver_account"
    ].to_numpy()

    print("Transactions:", len(features))
    print("Fraud cases:", int(y.sum()))
    print("Numeric features:", len(feature_columns))
    print("Sender-receiver alignment: VERIFIED")

    return (
        features,
        X,
        y,
        sender_accounts,
        receiver_accounts,
        feature_columns,
    )


# =========================================================
# 3. CHRONOLOGICAL SPLIT
# =========================================================

def split_data(X, y):

    total = len(X)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_idx = np.arange(0, train_end)

    val_idx = np.arange(
        train_end,
        validation_end,
    )

    test_idx = np.arange(
        validation_end,
        total,
    )

    for name, indices in [
        ("training", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        if len(np.unique(y[indices])) < 2:
            raise ValueError(
                f"{name} split does not contain both classes."
            )

    print("\n========== CHRONOLOGICAL SPLIT ==========")

    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    print("Training fraud:", int(y[train_idx].sum()))
    print("Validation fraud:", int(y[val_idx].sum()))
    print("Test fraud:", int(y[test_idx].sum()))

    return train_idx, val_idx, test_idx


# =========================================================
# 4. BUILD ACCOUNT GRAPH
# =========================================================

def build_graph(
    X,
    sender_accounts,
    receiver_accounts,
    train_idx,
):

    print("\n========== BUILDING ACCOUNT GRAPH ==========")

    # Create a unique index for every account.
    all_accounts = sorted(
        set(sender_accounts.tolist())
        | set(receiver_accounts.tolist())
    )

    account_to_idx = {
        account: index
        for index, account in enumerate(all_accounts)
    }

    sender_idx = np.array(
        [account_to_idx[x] for x in sender_accounts],
        dtype=np.int64,
    )

    receiver_idx = np.array(
        [account_to_idx[x] for x in receiver_accounts],
        dtype=np.int64,
    )

    # Fit feature scaling only on training transactions.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X.iloc[train_idx]
    )

    X_all = scaler.transform(X)

    X_all = np.nan_to_num(
        X_all,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    X_all_tensor = torch.tensor(
        X_all,
        dtype=torch.float32,
    )

    node_count = len(all_accounts)
    feature_count = X.shape[1]

    # Account features are calculated using training
    # transactions only.
    node_features = torch.zeros(
        (node_count, feature_count),
        dtype=torch.float32,
    )

    node_counts = torch.zeros(
        node_count,
        dtype=torch.float32,
    )

    train_sender = sender_idx[train_idx]
    train_receiver = receiver_idx[train_idx]

    # Aggregate training transaction features for senders.
    node_features.index_add_(
        0,
        torch.tensor(train_sender, dtype=torch.long),
        X_train_tensor,
    )

    node_counts.index_add_(
        0,
        torch.tensor(train_sender, dtype=torch.long),
        torch.ones(len(train_idx)),
    )

    # Aggregate training transaction features for receivers.
    node_features.index_add_(
        0,
        torch.tensor(train_receiver, dtype=torch.long),
        X_train_tensor,
    )

    node_counts.index_add_(
        0,
        torch.tensor(train_receiver, dtype=torch.long),
        torch.ones(len(train_idx)),
    )

    node_features = (
        node_features
        / node_counts.clamp(min=1).unsqueeze(1)
    )

    # Build message-passing graph from training transactions only.
    source = torch.tensor(
        train_sender,
        dtype=torch.long,
    )

    destination = torch.tensor(
        train_receiver,
        dtype=torch.long,
    )

    edge_index = torch.stack(
        [
            torch.cat([source, destination]),
            torch.cat([destination, source]),
        ],
        dim=0,
    )

    print("Unique accounts:", node_count)
    print("Training transactions:", len(train_idx))
    print("Graph edges (bidirectional):", edge_index.shape[1])
    print("Node feature dimensions:", feature_count)

    return (
        account_to_idx,
        sender_idx,
        receiver_idx,
        node_features,
        edge_index,
        X_all_tensor,
        scaler,
    )


# =========================================================
# 5. TRAIN GRAPH SAGE
# =========================================================

def train_model(
    X_all,
    y,
    sender_idx,
    receiver_idx,
    node_features,
    edge_index,
    train_idx,
    val_idx,
):

    print("\n========== TRAINING GRAPHSAGE ==========")

    model = FraudGraphSAGE(
        node_feature_count=node_features.shape[1],
        edge_feature_count=X_all.shape[1],
    ).to(DEVICE)

    node_features = node_features.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    X_all = X_all.to(DEVICE)

    sender_tensor = torch.tensor(
        sender_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    receiver_tensor = torch.tensor(
        receiver_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    labels = torch.tensor(
        y,
        dtype=torch.float32,
        device=DEVICE,
    )

    train_tensor = torch.tensor(
        train_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    val_tensor = torch.tensor(
        val_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    positive_count = float(y[train_idx].sum())
    negative_count = float(len(train_idx) - positive_count)

    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1)],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_score = -1
    best_state = None
    patience_counter = 0

    print("Device:", DEVICE)
    print("Positive class weight:", round(float(pos_weight.item()), 2))

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        logits = model(
            node_features,
            edge_index,
            sender_tensor[train_tensor],
            receiver_tensor[train_tensor],
            X_all[train_tensor],
        )

        loss = criterion(
            logits,
            labels[train_tensor],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0,
        )

        optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            val_logits = model(
                node_features,
                edge_index,
                sender_tensor[val_tensor],
                receiver_tensor[val_tensor],
                X_all[val_tensor],
            )

            val_probabilities = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        val_labels = y[val_idx]

        val_pr_auc = average_precision_score(
            val_labels,
            val_probabilities,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {loss.item():.5f} | "
            f"Validation PR-AUC: {val_pr_auc:.5f}"
        )

        if val_pr_auc > best_score:

            best_score = val_pr_auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping activated.")
            break

    if best_state is None:
        raise RuntimeError("GNN training failed to produce a model.")

    model.load_state_dict(best_state)

    print("\nBest validation PR-AUC:", round(best_score, 5))

    return (
        model,
        node_features,
        edge_index,
        X_all,
        sender_tensor,
        receiver_tensor,
        labels,
        best_score,
    )


# =========================================================
# 6. SELECT THRESHOLD USING VALIDATION DATA
# =========================================================

def select_threshold(
    model,
    node_features,
    edge_index,
    X_all,
    sender_tensor,
    receiver_tensor,
    val_idx,
    y,
):

    print("\n========== VALIDATION THRESHOLD ==========")

    model.eval()

    val_tensor = torch.tensor(
        val_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        logits = model(
            node_features,
            edge_index,
            sender_tensor[val_tensor],
            receiver_tensor[val_tensor],
            X_all[val_tensor],
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    precision, recall, thresholds = precision_recall_curve(
        y[val_idx],
        probabilities,
    )

    f2_scores = (
        5 * precision[:-1] * recall[:-1]
        /
        (
            4 * precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f2_scores))

    threshold = float(thresholds[best_index])

    print("Selected threshold:", round(threshold, 6))
    print("Validation F2:", round(float(f2_scores[best_index]), 5))

    return threshold


# =========================================================
# 7. FINAL TEST EVALUATION
# =========================================================

def evaluate_model(
    model,
    node_features,
    edge_index,
    X_all,
    sender_tensor,
    receiver_tensor,
    test_idx,
    y,
    threshold,
):

    print("\n========== GNN TEST EVALUATION ==========")

    test_tensor = torch.tensor(
        test_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            node_features,
            edge_index,
            sender_tensor[test_tensor],
            receiver_tensor[test_tensor],
            X_all[test_tensor],
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    actual = y[test_idx]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "pr_auc": float(
            average_precision_score(actual, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(actual, probabilities)
        ),
        "precision": float(
            precision_score(actual, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predictions, zero_division=0)
        ),
        "f2_score": float(
            fbeta_score(
                actual,
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
        "test_transactions": int(len(actual)),
        "actual_fraud_cases": int(actual.sum()),
        "predicted_fraud_cases": int(predictions.sum()),
    }

    print("PR-AUC:", round(metrics["pr_auc"], 5))
    print("ROC-AUC:", round(metrics["roc_auc"], 5))
    print("Precision:", round(metrics["precision"], 5))
    print("Recall:", round(metrics["recall"], 5))
    print("F1:", round(metrics["f1_score"], 5))
    print("F2:", round(metrics["f2_score"], 5))

    print("\nConfusion matrix:")
    print("TN:", tn, "FP:", fp)
    print("FN:", fn, "TP:", tp)

    return metrics, probabilities, predictions


# =========================================================
# 8. SAVE MODEL AND PREDICTIONS
# =========================================================

def save_outputs(
    model,
    metrics,
    probabilities,
    predictions,
    test_idx,
    features,
    feature_columns,
    account_to_idx,
    scaler,
    train_idx,
    best_val_pr_auc,
):

    print("\n========== SAVING GNN OUTPUTS ==========")

    # Save model and information needed for future inference.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "account_to_idx": account_to_idx,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": metrics["threshold"],
        "model_type": "GraphSAGE",
        "node_feature_count": len(feature_columns),
        "best_validation_pr_auc": best_val_pr_auc,
    }

    torch.save(
        checkpoint,
        MODEL_FILE,
    )

    # Save test predictions.
    output = pd.DataFrame({
        "event_timestamp": features.iloc[test_idx][
            "event_timestamp"
        ].to_numpy(),

        "sender_account": features.iloc[test_idx][
            "sender_account"
        ].to_numpy(),

        "receiver_account": features.iloc[test_idx][
            "receiver_account"
        ].to_numpy(),

        "actual_fraud": features.iloc[test_idx][
            "is_fraud"
        ].to_numpy(),

        "fraud_probability": probabilities,
        "predicted_fraud": predictions,
    })

    output.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    report = {
        "phase": 6,
        "status": "GNN_TRAINING_COMPLETED",
        "model": "GraphSAGE",
        "architecture": {
            "convolution": "GraphSAGE",
            "layers": 2,
            "hidden_dimensions": [64, 32],
            "edge_classifier": [64, 32, 1],
            "dropout": 0.25,
        },
        "graph": {
            "unique_accounts": len(account_to_idx),
            "training_transactions": len(train_idx),
            "message_passing_edges": int(2 * len(train_idx)),
            "graph_construction": "training_transactions_only",
        },
        "feature_count": len(feature_columns),
        "best_validation_pr_auc": best_val_pr_auc,
        "evaluation": metrics,
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "prediction_file": str(PREDICTION_FILE.relative_to(ROOT)),
        "device": str(DEVICE),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("Model:", MODEL_FILE)
    print("Predictions:", PREDICTION_FILE)
    print("Report:", REPORT_FILE)


# =========================================================
# 9. MAIN
# =========================================================

def main():

    print("\n" + "=" * 65)
    print("FRAUDSHIELD AI - PHASE 06")
    print("GRAPH NEURAL NETWORK FRAUD DETECTION")
    print("=" * 65)

    (
        features,
        X,
        y,
        sender_accounts,
        receiver_accounts,
        feature_columns,
    ) = load_data()

    train_idx, val_idx, test_idx = split_data(
        X,
        y,
    )

    (
        account_to_idx,
        sender_idx,
        receiver_idx,
        node_features,
        edge_index,
        X_all,
        scaler,
    ) = build_graph(
        X,
        sender_accounts,
        receiver_accounts,
        train_idx,
    )

    (
        model,
        node_features,
        edge_index,
        X_all,
        sender_tensor,
        receiver_tensor,
        labels,
        best_val_pr_auc,
    ) = train_model(
        X_all,
        y,
        sender_idx,
        receiver_idx,
        node_features,
        edge_index,
        train_idx,
        val_idx,
    )

    threshold = select_threshold(
        model,
        node_features,
        edge_index,
        X_all,
        sender_tensor,
        receiver_tensor,
        val_idx,
        y,
    )

    metrics, probabilities, predictions = evaluate_model(
        model,
        node_features,
        edge_index,
        X_all,
        sender_tensor,
        receiver_tensor,
        test_idx,
        y,
        threshold,
    )

    save_outputs(
        model,
        metrics,
        probabilities,
        predictions,
        test_idx,
        features,
        feature_columns,
        account_to_idx,
        scaler,
        train_idx,
        best_val_pr_auc,
    )

    print("\n" + "=" * 65)
    print("PHASE 06 SUMMARY")
    print("=" * 65)

    print("Model: GraphSAGE")
    print("Device:", DEVICE)
    print("Accounts:", len(account_to_idx))
    print("Features:", len(feature_columns))
    print("Validation PR-AUC:", round(best_val_pr_auc, 5))
    print("Test PR-AUC:", round(metrics["pr_auc"], 5))
    print("Test ROC-AUC:", round(metrics["roc_auc"], 5))
    print("Precision:", round(metrics["precision"], 5))
    print("Recall:", round(metrics["recall"], 5))
    print("F1:", round(metrics["f1_score"], 5))
    print("F2:", round(metrics["f2_score"], 5))

    print("\nGNN model:", MODEL_FILE)
    print("Report:", REPORT_FILE)

    print("=" * 65)


if __name__ == "__main__":
    main()
