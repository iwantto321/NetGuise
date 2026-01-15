# NetGuise: Early-Stage Protocol-Compliant Traffic Morphing

⚠️ **ANONYMITY WARNING**

This repository contains artifacts for a **double-blind submission to ACM CCS 2026**.  
Please maintain **author anonymity** during the review process.  
Do **not** distribute, attribute, or de-anonymize the authors.

---

## 📖 Overview

This repository contains the **official PyTorch implementation** of **NetGuise**,  
a generative framework designed to evade **early-stage ML-based censorship**  
while **strictly enforcing transport-layer protocol validity**.

NetGuise addresses the *feasibility gap* in censorship evasion by generating  
**packet-level, replayable traffic** that preserves TCP/TLS semantics.

**Paper Reference**

> *NetGuise: Early-Stage Protocol-Compliant Traffic Morphing against Black-Box Censors*

---

## ✨ Core Contributions

- **Evasion**  
  Achieves **>92% evasion rate** against state-of-the-art classifiers  
  (UniNet, FastFlow) — Paper §6.2

- **Validity**  
  Enforces **100% TCP/TLS compliance** via amortized optimization — Paper §4.3

- **Real-Time**  
  Less than **15 ms inference latency**, suitable for early-stage handshake morphing

---

## 📂 Repository Structure

```text
.
├── README.md
├── requirements.txt
├── input.pcap
├── Step1_data_processing.py
├── Step2_Train_censor.py
├── Step3_train_morpher.py
├── Step4_inference.py
├── Step5_validation.py
├── data/
│   ├── raw_samples/
│   └── processed/
└── models/
    ├── censors/
    └── morpher/
```

---

## 💾 Datasets (Detail & Availability)

We evaluate NetGuise across **three heterogeneous datasets**  
(Cellular, VPN, Residential) to test robustness.

### 1. Dataset Access

Due to the extreme size of raw PCAP repositories (>500GB), we provide  
**preprocessed feature matrices** sufficient to reproduce all results reported in the paper.

**Anonymous Review Link (Zenodo):**

```
[INSERT YOUR ANONYMOUS ZENODO LINK HERE]
```

**Contents:**

- UTMobileNet.zip  
- VANT.zip  
- VideoPF.zip  

Each archive includes `X.npy` (features) and `y.npy` (labels).

**Original Sources (Reference Only):**

- UTMobileNet-2021 — IEEE Dataport  
- VANT-2024 — Public Repository  

---

### 2. Dataset Composition (Paper Table 6)

#### A. UTMobileNet-2021 (Cellular / High Jitter)

| Application | Count | Application | Count |
|------------|------:|------------|------:|
| GoogleMaps | 3,184 | Pinterest | 1,272 |
| Netflix | 3,021 | Instagram | 1,259 |
| Reddit | 2,271 | Facebook | 1,238 |
| Spotify | 971 | Twitter | 682 |
| **Total Flows** | **~15,000** | | |

---

#### B. VANT-2024 (Encrypted VPN / IoT)

| Category | Count | Category | Count |
|---------|------:|---------|------:|
| File Transfer | 10,430 | Chat | 1,101 |
| C2 (Command) | 9,599 | VoIP | 617 |
| Streaming | 1,264 | YouTube | 1,031 |
| **Total Flows** | **~23,000** | | |

---

#### C. VideoPF-2025 (Adaptive Streaming)

| Application | Count | Application | Count |
|------------|------:|------------|------:|
| YouTube | 5,751 | Amazon | 4,525 |
| Netflix | 3,298 | Twitch | 1,066 |

---

## 🛠️ Setup & Requirements

We recommend using a fresh Conda environment to ensure reproducibility.

### 1. Environment Setup

```bash
conda create -n netguise python=3.9
conda activate netguise

pip install -r requirements.txt
```

---

### 2. System Dependencies (Required for Step 5)

```bash
# Ubuntu / Debian
sudo apt-get install tcpreplay tshark

# macOS
brew install tcpreplay wireshark
```

---

## 🚀 Reproduction Pipeline

A sample `input.pcap` is provided in the repository root for quick verification.

### Step 1: Feature Extraction (Paper §5.1)

```bash
python Step1_data_processing.py     --input_dir data/raw_samples     --save_dir data/processed     --workers 4
```

Ensure `input.pcap` is placed in `data/raw_samples/`.

---

### Step 2: Censor & Surrogate Training (Paper §6.1)

```bash
python Step2_Train_censor.py     --data_dir data/processed     --epochs 20     --save_dir models/censors
```

Outputs trained classifiers (e.g., `UniNet.pth`, `FastFlow.pth`).

---

### Step 3: Training NetGuise (Paper §4.3)

The generator minimizes the composite loss:

```math
L = L_adv + λ_feas L_feas
```

```bash
python Step3_train_morpher.py     --surrogate_path models/censors/UniNet.pth     --save_dir models/morpher     --epochs 20
```

---

### Step 4: Inference & Morphing (Paper §6.2)

```bash
python Step4_inference.py     --input data/raw_samples/input.pcap     --output results/morphed.pcap     --model_path models/morpher/netguise_morpher_optimized.pth
```

Produces a physically realizable morphed PCAP.

---

### Step 5: Validity Verification (Paper §6.4)

```bash
python Step5_validation.py     --real data/raw_samples/input.pcap     --morphed results/morphed.pcap
```

✔ Verifies TCP three-way handshakes  
✔ Checks checksum correctness  
✔ Outputs LaTeX tables for Paper Tables 9 and 10  

---

## 📜 License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.
