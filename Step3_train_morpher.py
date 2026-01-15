import os
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. SURROGATE (TRANSFORMER) =================

class TransformerModel(nn.Module):
    """
    The Surrogate Censor (UniNet Architecture).
    This model acts as the differentiable black-box proxy for the attacker.
    """

    def __init__(self, input_size=3, num_heads=4, ff_dim=128, num_classes=4):
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

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ================= 2. MORPHER (NETGUISE) =================

class NetGuiseMorpher(nn.Module):
    """
    NetGuise Generator: Learns the mapping from Source Flow -> Target Style.
    Capacity increased (128 dim, 8 heads) for better manifold learning.
    """

    def __init__(self, feat_dim=3, embed_dim=128, num_heads=8):
        super(NetGuiseMorpher, self).__init__()

        # Embedding Layers
        self.src_embed = nn.Linear(feat_dim, embed_dim)
        self.tgt_embed = nn.Linear(feat_dim, embed_dim)

        # Cross-Attention: Source attends to Target
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Self-Attention: Enforces temporal smoothness
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Decoder Head: Predicts deltas [Size, IAT]
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Constraint: Physical realizability (Non-negative deltas)
        self.relu = nn.ReLU()

    def forward(self, x_src, x_tgt):
        q = self.src_embed(x_src)
        k = self.tgt_embed(x_tgt)
        v = self.tgt_embed(x_tgt)

        # Cross Attention
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        x = self.norm1(q + attn_out)

        # Self Attention
        attn_out2, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + attn_out2)

        # Generate Deltas
        deltas = self.decoder(x)
        deltas = self.relu(deltas)

        # Reconstruct Flow
        morphed_size = x_src[:, :, 0] + deltas[:, :, 0]
        morphed_iat = x_src[:, :, 2] + deltas[:, :, 1]
        direction = x_src[:, :, 1]  # Direction is immutable

        x_morphed = torch.stack([morphed_size, direction, morphed_iat], dim=2)
        return x_morphed, deltas


# ================= 3. OPTIMIZED LOSS (CW LOSS) =================

class RobustNetGuiseLoss(nn.Module):
    """
    Carlini-Wagner (CW) Margin Loss + Operational Cost.
    L_adv: Ensures high-confidence evasion (Margin > Kappa).
    L_cost: Minimizes bandwidth and latency overhead.
    """

    def __init__(self, alpha=0.1, beta=0.1, kappa=0.0):
        super(RobustNetGuiseLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa  # Confidence Margin

    def forward(self, logits, target_labels, deltas):
        # --- 1. CW Margin Loss (L_adv) ---
        # Logic: Ensure Target Class score exceeds the second highest by at least Kappa.
        # This pushes the model towards high-confidence attacks better than CrossEntropy.

        batch_size = logits.shape[0]

        # Get Logits for the Target Class
        target_logits = logits[torch.arange(batch_size), target_labels]

        # Get the highest Logit among "Other" Classes
        # Mask Target Logit with negative infinity for Max calculation
        logits_copy = logits.clone()
        logits_copy[torch.arange(batch_size), target_labels] = -1e9
        max_other_logits, _ = torch.max(logits_copy, dim=1)

        # CW Loss Formula: max(0, max_other - target + kappa)
        # Loss is 0 only when Target > Max_Other + Kappa
        l_adv = torch.mean(torch.relu(max_other_logits - target_logits + self.kappa))

        # --- 2. Cost Loss (L_cost) ---
        l_size = torch.mean(deltas[:, :, 0])
        l_time = torch.mean(deltas[:, :, 1])
        l_cost = (self.alpha * l_size) + (self.beta * l_time)

        return l_adv + l_cost, l_adv, l_cost


# ================= 4. UTILS (DATA LOADING) =================

def load_data(data_dir: str, scaler_path: str, n_packets: int):
    try:
        X_full = np.load(os.path.join(data_dir, "X_data.npy"))
        y_full = np.load(os.path.join(data_dir, "y_label.npy"))
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        logging.error("Data files not found.")
        exit(1)

    X = X_full[:, :n_packets, :]
    N, T, F = X.shape

    # Scale Data
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F)

    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_full, dtype=torch.long), X_scaled, y_full


def precompute_pool_indices(pool_X: np.ndarray, pool_y: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Optimization: Pre-index pool by Class to accelerate matching.
    """
    pool_index = {}
    unique_classes = np.unique(pool_y)
    for c in unique_classes:
        c_idxs = np.where(pool_y == c)[0]
        pool_index[c] = c_idxs
    return pool_index


def get_valid_targets_optimized(y_src_batch, x_src_batch, pool_X, pool_y, pool_index_cache):
    """
    Selects target flows ensuring Direction Matching.
    """
    batch_size = y_src_batch.size(0)
    x_tgt_list = []
    y_tgt_list = []

    for i in range(batch_size):
        curr_y = y_src_batch[i].item()
        # Get direction signature (Sign) of the current sample
        src_dir_seq = np.sign(x_src_batch[i, :, 1].cpu().numpy())

        # Randomly select a Target Class different from the current one
        possible_classes = [k for k in pool_index_cache.keys() if k != curr_y]
        target_cls = random.choice(possible_classes)

        candidates = pool_index_cache[target_cls]

        # Greedy Search: Attempt to find a direction match
        # Strategy: If no exact match, pick the one with highest direction overlap
        best_idx = -1
        best_match_score = -1

        # Randomly sample 20 candidates for comparison (Speed trade-off)
        sample_indices = np.random.choice(candidates, min(20, len(candidates)), replace=False)

        for idx in sample_indices:
            tgt_dir_seq = np.sign(pool_X[idx, :, 1])
            if np.array_equal(src_dir_seq, tgt_dir_seq):
                best_idx = idx
                break  # Perfect match found
            else:
                # Calculate match score as fallback
                match_score = np.sum(src_dir_seq == tgt_dir_seq)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_idx = idx

        x_tgt_list.append(torch.tensor(pool_X[best_idx], dtype=torch.float32))
        y_tgt_list.append(target_cls)

    return torch.stack(x_tgt_list).to(DEVICE), torch.tensor(y_tgt_list, dtype=torch.long).to(DEVICE)


# ================= 5. MAIN TRAINING LOOP =================

def main():
    parser = argparse.ArgumentParser(description="Train NetGuise Morpher with CW Loss")
    parser.add_argument("--data_dir", type=str, default="../processed_dataset", help="Path to processed data")
    parser.add_argument("--model_dir", type=str, default="../saved_models", help="Path to surrogate models")
    parser.add_argument("--save_dir", type=str, default="../morpher_models", help="Path to save morpher")

    # Optimization: Lower default penalty weights to give the model more freedom
    parser.add_argument("--alpha", type=float, default=0.1, help="Bandwidth Cost Weight (Size)")
    parser.add_argument("--beta", type=float, default=0.1, help="Latency Cost Weight (IAT)")
    parser.add_argument("--kappa", type=float, default=1.0, help="CW Loss Confidence Margin")
    parser.add_argument("--epochs", type=int, default=100, help="Training Epochs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # A. Load Data
    logging.info("Loading Data...")
    scaler_path = os.path.join(args.model_dir, "scaler_attacker.pkl")
    X_train, y_train, X_pool_np, y_pool_np = load_data(args.data_dir, scaler_path, 30)
    pool_cache = precompute_pool_indices(X_pool_np, y_pool_np)  # Precompute indices

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    # B. Load Surrogate (Transformer)
    logging.info("Loading Surrogate Censor (Transformer)...")
    surrogate = TransformerModel().to(DEVICE)
    surrogate_path = os.path.join(args.model_dir, "Attacker_Transformer.pth")
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=DEVICE))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    # C. Setup Morpher & Optimized Loss
    # Initialize larger Morpher (128 dim)
    morpher = NetGuiseMorpher(embed_dim=128, num_heads=8).to(DEVICE)

    # Initialize CW Loss
    criterion = RobustNetGuiseLoss(alpha=args.alpha, beta=args.beta, kappa=args.kappa)

    # Slightly increase learning rate & Initialize Scheduler
    optimizer = optim.Adam(morpher.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # D. Train
    logging.info(f"Starting Optimized NetGuise Training (CW Loss, Kappa={args.kappa})...")

    for epoch in range(args.epochs):
        total_loss, total_asr = 0, 0
        total_adv_loss, total_cost_loss = 0, 0
        batches = 0

        for x_src, y_src in loader:
            x_src, y_src = x_src.to(DEVICE), y_src.to(DEVICE)

            # Optimized Target Selection
            x_tgt, y_tgt = get_valid_targets_optimized(y_src, x_src, X_pool_np, y_pool_np, pool_cache)

            optimizer.zero_grad()
            x_adv, deltas = morpher(x_src, x_tgt)
            logits = surrogate(x_adv)

            # Robust Loss
            loss, l_adv, l_cost = criterion(logits, y_tgt, deltas)

            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_adv_loss += l_adv.item()
            total_cost_loss += l_cost.item()

            preds = torch.argmax(logits, dim=1)
            total_asr += (preds == y_tgt).float().sum().item()
            batches += x_src.size(0)

        scheduler.step()  # Update LR

        avg_loss = total_loss / len(loader)
        avg_asr = total_asr / batches
        avg_l_adv = total_adv_loss / len(loader)
        avg_l_cost = total_cost_loss / len(loader)

        logging.info(
            f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} (Adv: {avg_l_adv:.4f}, Cost: {avg_l_cost:.4f}) | "
            f"ASR: {avg_asr * 100:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    torch.save(morpher.state_dict(), os.path.join(args.save_dir, "netguise_morpher_optimized.pth"))
    logging.info("Optimization Complete.")

    # E. Quick Eval
    morpher.eval()
    with torch.no_grad():
        test_x, test_y = next(iter(loader))
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
        target_x, target_y = get_valid_targets_optimized(test_y, test_x, X_pool_np, y_pool_np, pool_cache)

        morph_x, deltas = morpher(test_x, target_x)
        logits = surrogate(morph_x)
        preds = torch.argmax(logits, dim=1)

        asr = (preds == target_y).float().mean().item()
        pad = deltas[:, :, 0].mean().item()
        lat = deltas[:, :, 1].mean().item()

        print("\n=== Final Validation ===")
        print(f"ASR: {asr * 100:.2f}%")
        print(f"Avg Padding: {pad:.4f}")
        print(f"Avg Delay:   {lat:.4f}")


if __name__ == "__main__":
    main()