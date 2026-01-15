import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, IPv6, TCP, UDP
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Optional, Tuple, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_features_from_pcap(file_path: str, target_n: int = 100) -> Optional[np.ndarray]:
    """
    Parses a PCAP file to extract a feature matrix for the dominant flow.

    Args:
        file_path (str): Path to the .pcap or .pcapng file.
        target_n (int): Number of packets to extract per flow.

    Returns:
        np.ndarray: Feature matrix of shape (target_n, 3) containing [Size, Direction, IAT].
                    Returns None if the file is invalid or empty.
    """
    try:
        packets = rdpcap(file_path)
    except Exception as e:
        # Log warning but continue processing other files
        logging.debug(f"Failed to read {file_path}: {e}")
        return None

    if len(packets) == 0:
        return None

    # 1. Flow Aggregation
    # Dictionary to map canonical flow keys to packet lists
    flows = {}

    for pkt in packets:
        # Process only IP/IPv6 + TCP/UDP packets
        if (IP in pkt or IPv6 in pkt) and (TCP in pkt or UDP in pkt):

            # Extract layers
            ip_layer = pkt[IP] if IP in pkt else pkt[IPv6]
            trans_layer = pkt[TCP] if TCP in pkt else pkt[UDP]

            # Convert to appropriate types for comparison
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            src_port = int(trans_layer.sport)
            dst_port = int(trans_layer.dport)
            proto = "TCP" if TCP in pkt else "UDP"

            # Create a canonical key for bidirectional flows
            # Sort IPs (lexicographically) and Ports (numerically) to ensure
            # (A -> B) and (B -> A) map to the same key.
            if src_ip < dst_ip:
                key_ips = (src_ip, dst_ip)
            else:
                key_ips = (dst_ip, src_ip)

            if src_port < dst_port:
                key_ports = (src_port, dst_port)
            else:
                key_ports = (dst_port, src_port)

            # Key structure: (IP_Low, IP_High, Port_Low, Port_High, Proto)
            key = key_ips + key_ports + (proto,)

            if key not in flows:
                flows[key] = []
            flows[key].append(pkt)

    if not flows:
        return None

    # Select the main flow (the one with the most packets)
    main_flow_key = max(flows, key=lambda k: len(flows[k]))
    main_packets = flows[main_flow_key]

    # 2. Feature Extraction
    # Heuristic: Assume the first packet of the flow defines the 'Client' (initiator)
    first_pkt = main_packets[0]
    first_layer = first_pkt[IP] if IP in first_pkt else first_pkt[IPv6]
    client_ip = first_layer.src

    start_time = float(first_pkt.time)
    prev_time = start_time
    features = []

    for pkt in main_packets[:target_n]:
        # --- Feature 1: Packet Size ---
        if IP in pkt:
            size = pkt[IP].len
        elif IPv6 in pkt:
            # Approximate payload length + header
            size = pkt[IPv6].plen + 40
        else:
            size = len(pkt)

        # --- Feature 2: Direction ---
        # 1.0 = Outgoing (Client -> Server), -1.0 = Incoming (Server -> Client)
        pkt_src = pkt[IP].src if IP in pkt else pkt[IPv6].src
        direction = 1.0 if pkt_src == client_ip else -1.0

        # --- Feature 3: Inter-Arrival Time (IAT) ---
        curr_time = float(pkt.time)
        iat = curr_time - prev_time
        if iat < 0:
            iat = 0.0  # Handle potential timestamp anomalies
        prev_time = curr_time

        features.append([size, direction, iat])

    # 3. Padding / Truncation
    # Ensure the output is exactly (target_n, 3)
    features_np = np.array(features, dtype=np.float32)

    if len(features_np) < target_n:
        pad_len = target_n - len(features_np)
        padding = np.zeros((pad_len, 3), dtype=np.float32)
        features_np = np.vstack([features_np, padding])

    return features_np


def worker_func(args: Tuple[str, int, int]) -> Optional[Tuple[np.ndarray, int]]:
    """Wrapper function for multiprocessing."""
    file_path, label_idx, target_n = args
    data = extract_features_from_pcap(file_path, target_n)
    if data is not None:
        return data, label_idx
    return None


def generate_statistics(X: np.ndarray, y: np.ndarray, classes: List[str], save_path: str):
    """Generates and saves a CSV report on data distribution."""
    print("\n" + "=" * 20 + " DATA DISTRIBUTION REPORT " + "=" * 20)

    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))
    stats_data = []

    for idx, class_name in enumerate(classes):
        if idx not in count_dict:
            continue

        class_X = X[y == idx]
        n_samples = class_X.shape[0]

        # Calculate avg stats (ignoring zero-padding)
        # Use Packet Size column (index 0) to mask valid packets
        packet_sizes = class_X[:, :, 0].flatten()
        valid_mask = packet_sizes > 0

        valid_sizes = packet_sizes[valid_mask]
        avg_size = np.mean(valid_sizes) if len(valid_sizes) > 0 else 0

        all_iats = class_X[:, :, 2].flatten()
        valid_iats = all_iats[valid_mask]  # Use same mask
        avg_iat = np.mean(valid_iats) if len(valid_iats) > 0 else 0

        # Sum of IATs per flow = Total Duration
        flow_durations = np.sum(class_X[:, :, 2], axis=1)
        avg_duration = np.mean(flow_durations)

        stats_data.append({
            "Class": class_name,
            "Samples": n_samples,
            "Avg Pkt Size (B)": f"{avg_size:.2f}",
            "Avg IAT (s)": f"{avg_iat:.4f}",
            "Avg Duration (s)": f"{avg_duration:.2f}"
        })

    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string(index=False))

    report_file = os.path.join(save_path, "data_distribution.csv")
    df_stats.to_csv(report_file, index=False)
    logging.info(f"Statistics report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="PCAP Feature Extraction Tool")
    parser.add_argument("--data_root", type=str, default="../Dataset", help="Root directory of raw PCAP files")
    parser.add_argument("--save_dir", type=str, default="../processed_dataset",
                        help="Directory to save processed .npy files")
    parser.add_argument("--target_n", type=int, default=100, help="Number of packets to extract per flow")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    # Define classes based on directory structure expectation or hardcoded list
    # Assuming subfolders in data_root correspond to classes
    # If strictly using the list from original code:
    CLASSES = ["amazon", "youtube", "twitch", "netflix"]

    if not os.path.exists(args.data_root):
        logging.error(f"Data root not found: {args.data_root}")
        return

    logging.info(f"Starting processing with {args.workers} workers...")
    logging.info(f"Target Packets: {args.target_n} | Classes: {CLASSES}")

    # 1. Scan Files
    all_files = []
    for idx, class_name in enumerate(CLASSES):
        folder_path = os.path.join(args.data_root, class_name)
        if not os.path.exists(folder_path):
            logging.warning(f"Class folder not found: {folder_path}")
            continue

        # Support both .pcap and .pcapng
        files = glob.glob(os.path.join(folder_path, "*.pcap")) + \
                glob.glob(os.path.join(folder_path, "*.pcapng"))

        logging.info(f"Found {len(files)} files for class '{class_name}'")
        for f in files:
            # Pass target_n to the worker
            all_files.append((f, idx, args.target_n))

    if not all_files:
        logging.error("No PCAP files found. Exiting.")
        return

    # 2. Parallel Processing
    X_list = []
    y_list = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use tqdm to display progress bar
        results = list(tqdm(executor.map(worker_func, all_files), total=len(all_files), desc="Extracting Features"))

    # 3. Aggregate Results
    for res in results:
        if res is not None:
            feat, label = res
            X_list.append(feat)
            y_list.append(label)

    if not X_list:
        logging.error("No valid features extracted from files.")
        return

    X = np.array(X_list)  # Shape: (Total_Samples, target_n, 3)
    y = np.array(y_list)  # Shape: (Total_Samples, )

    logging.info("Processing Complete!")
    logging.info(f"Final Dataset Shape: X={X.shape}, y={y.shape}")

    # 4. Save Data
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "X_data.npy"), X)
    np.save(os.path.join(args.save_dir, "y_label.npy"), y)
    np.save(os.path.join(args.save_dir, "classes.npy"), CLASSES)
    logging.info(f"Numpy arrays saved to {args.save_dir}")

    # 5. Generate Statistics
    generate_statistics(X, y, CLASSES, args.save_dir)


if __name__ == "__main__":
    main()