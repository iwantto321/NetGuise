# NetGuise: Early-Stage Protocol-Compliant Traffic Morphing

[![Submission](https://img.shields.io/badge/Submission-ACM%20CCS%202026-blue)](https://www.sigsac.org/ccs/CCS2026/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> **⚠️ ANONYMITY WARNING:** This repository contains artifacts for a double-blind submission to ACM CCS 2026. Please maintain author anonymity during the review process.

This repository contains the official PyTorch implementation of **NetGuise**, a generative framework designed to evade early-stage ML-based censorship while strictly enforcing protocol validity.

## 📖 Overview

NetGuise addresses the "Feasibility Gap" in censorship evasion by generating **packet-level, replayable traffic**.
* **Paper Reference:** *NetGuise: Early-Stage Protocol-Compliant Traffic Morphing against Black-Box Censors*
* **Core Capabilities:**
    1.  **Evasion:** >92% evasion rate against SOTA classifiers (UniNet, FastFlow).
    2.  **Validity:** 100% TCP/TLS compliance via amortized optimization.
    3.  **Real-Time:** <15ms inference latency.

## 📂 Repository Structure

The codebase is organized into 5 sequential steps mapping to the paper's pipeline:

```text
.
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── input.pcap                  # Sample raw traffic (VideoPF subset)
├── Step1_data_processing.py    # Feature extraction (Paper §5.1)
├── Step2_Train_censor.py       # Censor/Surrogate training (Paper §6.1)
├── Step3_train_morpher.py      # NetGuise Generator training (Paper §4)
├── Step4_inference.py          # Adversarial trace generation (Paper §6.2)
├── Step5_validation.py         # Validity & Replay checks (Paper §6.4)
├── data/                       # Dataset directory
│   ├── raw_samples/            # Place raw PCAPs here
│   └── processed/              # Extracted features (.npy)
└── models/                     # Saved checkpoints
    ├── censors/                # Pre-trained classifiers
    └── morpher/                # Trained NetGuise generator
💾 Datasets (Detail & Download)This study utilizes three distinct datasets to evaluate robustness across different network conditions (Cellular, VPN, Residential).1. Dataset AvailabilityDue to the large size (>500GB) of the raw PCAP repositories, we provide preprocessed feature matrices (.npy) which are sufficient to reproduce all training and evasion results.🔗 Download Full Preprocessed Datasets (Review Link):[Zenodo Anonymous Link Placeholder] (Contains UTMobileNet.zip, VANT.zip, VideoPF.zip)🔗 Original Sources (Reference Only):UTMobileNet-2021: [IEEE Dataport Link / Public Repo]VANT-2024: [Public Repo Link]2. Dataset Composition (Paper Table 6)Detailed class distribution for the datasets used in our evaluation:A. UTMobileNet-2021 (Cellular/High-Jitter)Description: Captures real-world mobile traffic with significant jitter and packet loss.| Application | Count | Application | Count || :--- | :--- | :--- | :--- || GoogleMaps | 3,184 | Pinterest | 1,272 || Netflix | 3,021 | Instagram | 1,259 || Reddit | 2,271 | Facebook | 1,238 || Total Flows | ~15,000 | Metric | T-ESR |B. VANT-2024 (Encrypted VPN/IoT)Description: Encrypted VPN tunnel traffic representing IoT and generic secure channels.| Category | Count | Category | Count || :--- | :--- | :--- | :--- || File Transfer | 10,430 | Chat | 1,101 || C2 (Command) | 9,599 | VoIP | 617 || Streaming | 1,264 | | || Total Flows | ~23,000 | | |C. VideoPF-2025 (Adaptive Streaming)Description: Custom dataset capturing adaptive-bitrate (ABR) video streaming under realistic shaping.| Application | Count | Application | Count || :--- | :--- | :--- | :--- || YouTube | 5,751 | Amazon | 4,525 || Netflix | 3,298 | Twitch | 1,066 |🛠️ Setup & RequirementsWe recommend using a Conda environment.Bash# 1. Create Environment
conda create -n netguise python=3.9
conda activate netguise

# 2. Install Python Dependencies
pip install -r requirements.txt

# 3. Install System Tools (Required for Step 5 Validation)
# Ubuntu/Debian
sudo apt-get install tcpreplay tshark
# MacOS
brew install tcpreplay wireshark
🚀 Reproduction PipelineWe provide a sample input.pcap in the root directory to allow immediate testing of the pipeline.Step 1: Feature Extraction (Paper §5.1)Extracts flow features (Packet Size, Direction, IAT) from raw PCAP traces.Bashpython Step1_data_processing.py \
    --input_dir data/raw_samples \
    --save_dir data/processed \
    --workers 4
Note: Place input.pcap into data/raw_samples/ before running.Step 2: Censor & Surrogate Training (Paper §6.1)Trains the baseline censors (Grad-BP, CNN, LSTM, UniNet) and the surrogate model used by the attacker.Bashpython Step2_Train_censor.py \
    --data_dir data/processed \
    --epochs 20 \
    --save_dir models/censors
Output: Saves trained models to models/censors/.Step 3: Training NetGuise (Paper §4.3)Trains the amortized generator using the differentiable surrogate.Bashpython Step3_train_morpher.py \
    --surrogate_path models/censors/UniNet_Transformer.pth \
    --save_dir models/morpher \
    --epochs 20
Output: Saves netguise_morpher_optimized.pth. This implements the composite loss $\mathcal{L} = \mathcal{L}_{adv} + \lambda_{feas}\mathcal{L}_{feas}$.Step 4: Inference & Morphing (Paper §6.2)Generates adversarial traces by applying the learned morphing policy.Bashpython Step4_inference.py \
    --input data/raw_samples/input.pcap \
    --output results/morphed.pcap \
    --model_path models/morpher/netguise_morpher_optimized.pth
Output: A physically realizable .pcap file.Step 5: Validity Verification (Paper §6.4)[Highlight] Automatically verifies protocol compliance and generates LaTeX tables matching the paper.Bashpython Step5_validation.py \
    --real data/raw_samples/input.pcap \
    --morphed results/morphed.pcap
Output: Prints LaTeX code for Table 9 (Sensitivity) and Table 10 (Replay Results) to the console.📜 LicenseThis project is licensed under the MIT License. See LICENSE for details.
