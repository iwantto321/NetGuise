
# NetGuise: Protocol-Compliant Traffic Morphing against Black-Box Censors

[![Artifact Evaluation](https://img.shields.io/badge/Artifact-Available-green.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Submission for CCS (Anonymous)**
> This repository contains the reference implementation of **NetGuise**, a system for real-time, protocol-compliant censorship evasion.

## 📂 Project Structure

- `src/models.py`: Core PyTorch definitions for the Morpher and Surrogate Censor (UniNet).
- `src/1_preprocess.py`: Extracts flow features from raw PCAP files.
- `src/2_train_censors.py`: Trains the target and surrogate classifiers.
- `src/3_train_morpher.py`: Optimizes the NetGuise Morpher using CW Loss.
- `src/4_inference.py`: End-to-end pipeline to morph real traffic traces.
- `src/5_validate.py`: Verifies TCP semantics and generates evaluation tables.

## 🚀 Quick Start (Reproducibility)

We provide a shell script to run the full pipeline on a sample trace.

### 1. Requirements
```bash
pip install -r requirements.txt


# Step 1: Preprocess Data (Assuming ./data/raw exists)
python src/1_preprocess.py --data_root ./data/raw --save_dir ./data/processed

# Step 2: Train Censors (Target & Surrogate)
python src/2_train_censors.py --data_dir ./data/processed --save_dir ./saved_models

# Step 3: Train NetGuise Morpher
python src/3_train_morpher.py --model_dir ./saved_models --epochs 30

# Step 4: Inference (Morph a trace)
python src/4_inference.py --input ./data/test_trace.pcap --output ./data/morphed.pcap

# Step 5: Validate Results (Generate Table 9 & 10)
python src/5_validate.py --morphed ./data/morphed.pcap --real ./data/test_trace.pcap
