import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scapy.all import rdpcap, wrpcap, Ether, IP, TCP, UDP, Raw
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30  # Must match N_PACKETS in training
FEATURE_DIM = 3
MTU = 1500


# ================= 1. MODEL DEFINITIONS (Must match training) =================

class TransformerModel(nn.Module):
    """
    The Surrogate Censor (Transformer / UniNet Architecture).
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


class NetGuiseMorpher(nn.Module):
    """
    NetGuise Generator (Amortized Optimization).
    """

    def __init__(self, feat_dim=3, embed_dim=128, num_heads=8):
        super(NetGuiseMorpher, self).__init__()
        self.src_embed = nn.Linear(feat_dim, embed_dim)
        self.tgt_embed = nn.Linear(feat_dim, embed_dim)

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Decoder: [Delta_Size, Delta_IAT]
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.relu = nn.ReLU()

    def forward(self, x_src, x_tgt):
        q = self.src_embed(x_src)
        k = self.tgt_embed(x_tgt)
        v = self.tgt_embed(x_tgt)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        x = self.norm1(q + attn_out)

        attn_out2, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + attn_out2)

        deltas = self.decoder(x)
        deltas = self.relu(deltas)

        morphed_size = x_src[:, :, 0] + deltas[:, :, 0]
        morphed_iat = x_src[:, :, 2] + deltas[:, :, 1]
        direction = x_src[:, :, 1]

        x_morphed = torch.stack([morphed_size, direction, morphed_iat], dim=2)
        return x_morphed, deltas


# ================= 2. PIPELINE CORE =================

class MorphingPipeline:
    def __init__(self, censor_path, morpher_path, scaler_path):
        self.device = DEVICE
        logging.info(f"Initializing Pipeline on {self.device}...")

        try:
            # 1. Load Surrogate (Transformer)
            self.censor = TransformerModel().to(self.device)
            # Use weights_only=True for security if using newer PyTorch, else standard load
            try:
                state_dict = torch.load(censor_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(censor_path, map_location=self.device)  # Fallback for older torch

            self.censor.load_state_dict(state_dict)
            self.censor.eval()

            # 2. Load Morpher (NetGuise)
            self.morpher = NetGuiseMorpher().to(self.device)
            try:
                state_dict = torch.load(morpher_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(morpher_path, map_location=self.device)

            self.morpher.load_state_dict(state_dict)
            self.morpher.eval()

            # 3. Load Scaler
            self.scaler = joblib.load(scaler_path)
            logging.info("Models & Scaler loaded successfully.")

        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            logging.warning("Running in DUMMY MODE (Random Weights). Do not use for real evaluation.")
            self.scaler = None
            # Initialize with random weights for demo flow
            self.censor = TransformerModel().to(self.device)
            self.morpher = NetGuiseMorpher().to(self.device)

    def get_traffic_summary(self, packets, label="Unknown"):
        if not packets: return {}

        sizes = [len(p) for p in packets]
        times = [float(p.time) for p in packets]
        duration = times[-1] - times[0] if len(times) > 1 else 0.001

        stats = {
            "Label": label,
            "Count": len(packets),
            "Bytes": sum(sizes),
            "Dur(s)": round(duration, 3),
            "AvgSz": round(np.mean(sizes), 1),
            "AvgIAT": round(np.mean(np.diff(times)), 4) if len(times) > 1 else 0,
            "Kbps": round(sum(sizes) * 8 / duration / 1000, 1)
        }
        return stats

    def extract_features(self, packets):
        """Extracts (Seq, 3) features from packet list."""
        logging.info(f"Extracting features from {len(packets)} packets...")
        feats = []
        if not packets: return None, None

        base_time = float(packets[0].time)
        # Identify Client IP (Heuristic: Src of first packet)
        if IP in packets[0]:
            client_ip = packets[0][IP].src
        elif IPv6 in packets[0]:
            client_ip = packets[0][IPv6].src
        else:
            client_ip = None  # Fallback

        limit = min(len(packets), SEQ_LEN)

        for i in range(limit):
            p = packets[i]
            if IP not in p and IPv6 not in p: continue

            # 1. Size
            if IP in p:
                size = p[IP].len
            elif IPv6 in p:
                size = p[IPv6].plen + 40
            else:
                size = len(p)

            # 2. Direction
            # +1: Outgoing (Client->Server), -1: Incoming (Server->Client)
            if client_ip:
                p_src = p[IP].src if IP in p else p[IPv6].src
                direction = 1.0 if p_src == client_ip else -1.0
            else:
                direction = 1.0

            # 3. IAT
            if i == 0:
                iat = 0.0
            else:
                iat = float(p.time) - float(packets[i - 1].time)
                if iat < 0: iat = 0.0  # Fix timestamp jitter

            feats.append([size, direction, iat])

        # Padding
        while len(feats) < SEQ_LEN:
            feats.append([0, 0, 0])

        feats_np = np.array(feats)

        # Scaling
        if self.scaler:
            N, F = feats_np.shape
            feats_scaled = self.scaler.transform(feats_np).reshape(1, N, F)
        else:
            feats_scaled = feats_np.reshape(1, SEQ_LEN, FEATURE_DIM)

        return torch.tensor(feats_scaled, dtype=torch.float32).to(self.device), feats_np

    def detect_traffic(self, x_tensor):
        """Runs the Surrogate Censor to predict traffic class."""
        if x_tensor is None: return -1

        with torch.no_grad():
            logits = self.censor(x_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

        logging.info(f"Surrogate Prediction: Class {pred} (Conf: {conf:.2f})")
        return pred

    def generate_morphing_plan(self, x_src):
        """Generates Deltas (Padding, Delay) using NetGuise."""
        if x_src is None: return None

        # Create a Dummy Target (In real use, load from a target pcap)
        # Here we simulate "Impersonating Class 1" by adding synthetic noise style
        x_tgt = x_src.clone()
        # Heuristic: Target has slightly larger packets and smaller IAT
        x_tgt[:, :, 0] += 0.5
        x_tgt[:, :, 2] -= 0.2

        with torch.no_grad():
            _, deltas = self.morpher(x_src, x_tgt)

        # Deltas: (1, 30, 2)
        deltas_np = deltas.cpu().numpy()[0]

        # Denormalize (Approximate)
        # Assuming Scaler was fitted on standard internet traffic
        # Scale factors are empirical approximations of StdDev
        SCALE_SIZE = 500.0  # 1 unit ~= 500 bytes
        SCALE_TIME = 0.1  # 1 unit ~= 100 ms

        real_deltas = deltas_np.copy()
        real_deltas[:, 0] *= SCALE_SIZE
        real_deltas[:, 1] *= SCALE_TIME

        logging.info(
            f"Generated Plan -> Avg Pad: {np.mean(real_deltas[:, 0]):.1f}B, Avg Dly: {np.mean(real_deltas[:, 1]):.3f}s")
        return real_deltas

    def apply_modifications(self, original_packets, deltas):
        """Applies padding and delays to the PCAP using Scapy."""
        logging.info("Injecting perturbations into trace...")
        morphed_pkts = []
        if not original_packets: return []

        base_time = float(original_packets[0].time)
        current_time = base_time

        for i, pkt in enumerate(original_packets):
            pkt = pkt.copy()  # Critical: Don't modify original list

            # Only IP/TCP/UDP supported for now
            if not (IP in pkt or IPv6 in pkt):
                morphed_pkts.append(pkt)
                continue

            # Modify only first SEQ_LEN packets
            if i < len(deltas):
                pad_bytes = int(deltas[i][0])
                delay_sec = float(deltas[i][1])

                # 1. Apply Padding (MTU Aware)
                if pad_bytes > 0:
                    # Check current size
                    curr_len = len(pkt)
                    available_space = MTU - curr_len

                    if available_space > 0:
                        actual_pad = min(pad_bytes, available_space)
                        padding_load = b'\x00' * actual_pad

                        if Raw in pkt:
                            pkt[Raw].load += padding_load
                        else:
                            pkt = pkt / Raw(load=padding_load)

                        # Fix headers
                        if IP in pkt:
                            del pkt[IP].len
                            del pkt[IP].chksum
                        if TCP in pkt:
                            del pkt[TCP].chksum
                        if UDP in pkt:
                            del pkt[UDP].len
                            del pkt[UDP].chksum

                # 2. Apply Delay (Causal)
                # Ensure strictly non-negative relative to previous packet
                if i > 0:
                    prev_pkt_time = float(morphed_pkts[-1].time)
                    # Original gap
                    orig_iat = float(original_packets[i].time) - float(original_packets[i - 1].time)
                    if orig_iat < 0: orig_iat = 0

                    new_iat = orig_iat + delay_sec
                    current_time = prev_pkt_time + new_iat
                    pkt.time = current_time
                else:
                    pkt.time = base_time  # First packet anchor

            else:
                # Packets beyond window: maintain original relative timing
                if i > 0:
                    orig_iat = float(original_packets[i].time) - float(original_packets[i - 1].time)
                    current_time += orig_iat
                    pkt.time = current_time

            morphed_pkts.append(pkt)

        return morphed_pkts


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="NetGuise Inference Engine")
    parser.add_argument("--input", type=str, default="input.pcap", help="Input PCAP file")
    parser.add_argument("--output", type=str, default="morphed_output.pcap", help="Output PCAP file")
    parser.add_argument("--model_dir", type=str, default="../pipeline_models", help="Model directory")
    args = parser.parse_args()

    # Define Paths
    path_surrogate = os.path.join(args.model_dir, "Attacker_Transformer.pth")
    path_morpher = os.path.join(args.model_dir, "netguise_morpher_optimized.pth")  # Or 'netguise_morpher.pth'
    path_scaler = os.path.join(args.model_dir, "scaler_attacker.pkl")  # Assuming we use Attacker's scaler

    # 1. Check Input
    if not os.path.exists(args.input):
        logging.info(f"Input {args.input} not found. Generating dummy traffic...")
        # Generate dummy TCP handshake + data
        pkts = []
        t = time.time()
        # Handshake
        pkts.append(IP(dst="1.2.3.4") / TCP(flags="S"));
        pkts[-1].time = t
        pkts.append(IP(src="1.2.3.4") / TCP(flags="SA"));
        pkts[-1].time = t + 0.02
        pkts.append(IP(dst="1.2.3.4") / TCP(flags="A"));
        pkts[-1].time = t + 0.021
        # Data
        for k in range(20):
            pkts.append(IP(dst="1.2.3.4") / TCP(flags="PA") / Raw(load=f"Payload_{k}" * 10))
            pkts[-1].time = t + 0.05 + (k * 0.1)
        wrpcap(args.input, pkts)

    # 2. Init Pipeline
    pipeline = MorphingPipeline(path_surrogate, path_morpher, path_scaler)

    # 3. Process
    logging.info("-" * 40)
    packets = rdpcap(args.input)
    stats_orig = pipeline.get_traffic_summary(packets, "Original")

    # A. Feature Extraction
    x_tensor, _ = pipeline.extract_features(packets)

    # B. Initial Detection
    cls_orig = pipeline.detect_traffic(x_tensor)

    # C. Morphing
    deltas = pipeline.generate_morphing_plan(x_tensor)
    morphed_packets = pipeline.apply_modifications(packets, deltas)

    # D. Save & Verify
    if morphed_packets:
        wrpcap(args.output, morphed_packets)
        stats_morph = pipeline.get_traffic_summary(morphed_packets, "Morphed")

        # Verify
        x_new, _ = pipeline.extract_features(morphed_packets)
        cls_new = pipeline.detect_traffic(x_new)

        print("\n" + "=" * 40)
        print("PERFORMANCE SUMMARY")
        print("=" * 40)
        df = pd.DataFrame([stats_orig, stats_morph])
        print(df.to_string(index=False))
        print("-" * 40)

        if cls_new != cls_orig:
            print(f"✅ SUCCESS: Evasion Successful ({cls_orig} -> {cls_new})")
        else:
            print(f"⚠️  WARNING: Failed to evade (Still {cls_orig}). Check Morpher weights.")
    else:
        logging.error("Morphing failed (Empty packet list).")


if __name__ == "__main__":
    main()