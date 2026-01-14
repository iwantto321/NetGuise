import os
import argparse
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. DATASET DEFINITION =================

class TrafficDataset(Dataset):
    """Custom PyTorch Dataset for traffic traces."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ================= 2. MODEL DEFINITIONS =================

class CNNModel(nn.Module):
    """
    1D-CNN for spatial feature extraction (inspired by Deep Fingerprinting).
    Input Shape: (Batch, Seq_Len, Features) -> Permuted internally to (Batch, Features, Seq_Len)
    """

    def __init__(self, input_len: int = 30, num_features: int = 3, num_classes: int = 4):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Global Pooling to handle variable lengths robustly
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.fc(x)


class LSTMModel(nn.Module):
    """
    LSTM-based classifier (FastFlow-Pkt architecture).
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_classes: int = 4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq, Feat)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_step = lstm_out[:, -1, :]
        x = self.dropout(last_step)
        return self.fc(x)


class TransformerModel(nn.Module):
    """
    Transformer-Encoder based classifier (UniNet-Pkt architecture).
    """

    def __init__(self, input_size: int = 3, num_heads: int = 4, ff_dim: int = 128, num_classes: int = 4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        # Permute for pooling: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ================= 3. TRAINING UTILS =================

def train_torch_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model_name: str,
                      epochs: int = 40) -> nn.Module:
    """
    Generic training loop for PyTorch models with Early Stopping.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None
    patience_counter = 0
    patience_limit = 5

    logging.info(f"  Starting training for {model_name} on {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        acc = accuracy_score(labels, preds)

        # Early Stopping Logic
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            logging.info(f"  [{model_name}] Early stopping at epoch {epoch + 1} | Best Acc: {best_acc:.4f}")
            break

    model.load_state_dict(best_state)
    return model


# ================= 4. MAIN WORKFLOW =================

def main():
    parser = argparse.ArgumentParser(description="Train Censor Models (Target & Surrogate)")
    parser.add_argument("--data_dir", type=str, default="../processed_dataset", help="Path to processed .npy files")
    parser.add_argument("--save_dir", type=str, default="../saved_models", help="Directory to save trained models")
    parser.add_argument("--split_data_dir", type=str, default="./split_data",
                        help="Directory to save split datasets (test set)")
    parser.add_argument("--n_packets", type=int, default=30, help="Number of packets to use for classification")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.split_data_dir, exist_ok=True)

    # --- A. Data Loading ---
    logging.info("Loading dataset...")
    try:
        X_full = np.load(os.path.join(args.data_dir, "X_data.npy"))
        y_full = np.load(os.path.join(args.data_dir, "y_label.npy"))
        classes = np.load(os.path.join(args.data_dir, "classes.npy"))
    except FileNotFoundError as e:
        logging.error(f"Dataset files not found in {args.data_dir}. Run preprocessing first.")
        return

    num_classes = len(classes)

    # Truncate to N_PACKETS (Shape: Samples, N, Features)
    X_truncated = X_full[:, :args.n_packets, :]
    logging.info(f"Data Loaded. Shape: {X_truncated.shape}")

    # --- B. Dataset Splitting ---
    # 1. Hold-out Test Set (20%) - Used for final table evaluation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_truncated, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    # 2. Split remaining (80%) into Defender (Target) and Attacker (Surrogate)
    # Ratio: Defender gets 50% of total, Attacker gets 30% of total.
    # From X_temp (80%), we need 5/8 for Defender (50%) and 3/8 for Attacker (30%).
    X_defender, X_attacker, y_defender, y_attacker = train_test_split(
        X_temp, y_temp, train_size=0.625, stratify=y_temp, random_state=42
    )

    logging.info(
        f"Split Sizes -> Defender (Target): {X_defender.shape[0]} | Attacker (Surrogate): {X_attacker.shape[0]} | Test: {X_test.shape[0]}")

    # Save Test Data for Evaluation Script
    np.save(os.path.join(args.split_data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.split_data_dir, "y_test.npy"), y_test)

    # --- C. Scaling ---
    # Important: Scale Defender and Attacker independently to simulate black-box scenario

    # Scale Defender
    scaler_def = StandardScaler()
    N, T, F = X_defender.shape
    X_def_scaled = scaler_def.fit_transform(X_defender.reshape(-1, F)).reshape(N, T, F)
    joblib.dump(scaler_def, os.path.join(args.save_dir, "scaler_defender.pkl"))

    # Scale Attacker
    scaler_att = StandardScaler()
    N_att, T_att, F_att = X_attacker.shape
    X_att_scaled = scaler_att.fit_transform(X_attacker.reshape(-1, F_att)).reshape(N_att, T_att, F_att)
    joblib.dump(scaler_att, os.path.join(args.save_dir, "scaler_attacker.pkl"))

    # --- D. Training Loop ---
    # We train two sets of models:
    # 1. Defender Models: Represent the real-world censors (Evaluation targets).
    # 2. Attacker Models: Represent the local surrogates (Used to train NetGuise).

    tasks = [
        ("Defender", X_def_scaled, y_defender),
        ("Attacker", X_att_scaled, y_attacker)
    ]

    for role, X_data, y_data in tasks:
        logging.info(f"\n{'=' * 20} Training {role} Models {'=' * 20}")

        # Internal Train/Val Split
        X_tr, X_val, y_tr, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

        train_loader = DataLoader(TrafficDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TrafficDataset(X_val, y_val), batch_size=args.batch_size)

        # 1. SVM (Baseline / Non-DL Censor)
        logging.info(f"Training {role} SVM...")
        svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        # Flatten for SVM: (Batch, Seq*Feat)
        svm.fit(X_tr.reshape(X_tr.shape[0], -1), y_tr)
        svm_acc = svm.score(X_val.reshape(X_val.shape[0], -1), y_val)
        logging.info(f"  [{role} SVM] Validation Acc: {svm_acc:.4f}")
        joblib.dump(svm, os.path.join(args.save_dir, f"{role}_SVM.pkl"))

        # 2. CNN (Deep Fingerprinting Style)
        logging.info(f"Training {role} CNN...")
        cnn = CNNModel(input_len=args.n_packets, num_classes=num_classes)
        cnn = train_torch_model(cnn, train_loader, val_loader, f"{role} CNN", args.epochs)
        torch.save(cnn.state_dict(), os.path.join(args.save_dir, f"{role}_CNN.pth"))

        # 3. LSTM (FastFlow Style)
        logging.info(f"Training {role} LSTM...")
        lstm = LSTMModel(input_size=3, num_classes=num_classes)
        lstm = train_torch_model(lstm, train_loader, val_loader, f"{role} LSTM", args.epochs)
        torch.save(lstm.state_dict(), os.path.join(args.save_dir, f"{role}_LSTM.pth"))

        # 4. Transformer (UniNet Style)
        logging.info(f"Training {role} Transformer...")
        tf_model = TransformerModel(input_size=3, num_classes=num_classes)
        tf_model = train_torch_model(tf_model, train_loader, val_loader, f"{role} Transformer", args.epochs)
        torch.save(tf_model.state_dict(), os.path.join(args.save_dir, f"{role}_Transformer.pth"))

    logging.info(f"\nAll models saved to {args.save_dir}")
    logging.info("Training pipeline complete.")


if __name__ == "__main__":
    main()