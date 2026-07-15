# NetGuise

NetGuise is a target-conditioned traffic morphing framework for evaluating
early-stage encrypted traffic classifiers. It generates packet-level padding
and non-negative delay actions while preserving packet direction, order, and
count under explicit MTU, bandwidth, and latency constraints.

This repository contains research code accompanying the NetGuise manuscript.

## What is implemented

- A 30-packet early-stage representation with packet size, direction, and
  inter-arrival time.
- Morphability-aware target-reference selection within a fixed target class.
- A shared source/target Transformer encoder and target-conditioned
  cross-attention decoder.
- Sigmoid action projection into the remaining padding and delay budgets.
- Full-window inference and a masked-causal online mode. Online mode hides all
  future packet features, commits only the current action, and never revises a
  committed action.
- Adapted FastFlow and UniNet classifier baselines for the early-flow feature
  space used here.
- Constraint validation for generated feature plans.

The FastFlow and UniNet models are paper-informed adaptations, not copies of
the authors' complete upstream implementations. FastFlow uses packet and
time-interval LSTM branches with confidence-based early stopping. UniNet uses
packet tokens, a flow-summary token, segment embeddings, learnable positional
embeddings, and a lightweight pre-normalized Transformer encoder.

## Repository layout

```text
netguise/                    Core package
  cli/                       Command-line workflows
  models/                    Morpher and classifier definitions
scripts/                     Stable artifact entry points
configs/default.json         Paper-aligned default parameters
tests/                       Unit and smoke tests
data/                        Example PCAP and processed features
models/                      Checkpoints and model notes
TestBed-live-SDN/            Controlled network testbed
ARTIFACT.md                  Reproduction and verification guide
```

The `Step1_...py` through `Step5_...py` files remain as small compatibility
entry points. New work should use `scripts/` or the installed console commands.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the test suite after installation:

```bash
pytest
```

## Reproduction workflow

The included processed dataset contains 100 packets per flow for compatibility
with earlier experiments. Training commands use the first 30 packets by
default.

### 1. Optional PCAP preprocessing

Raw PCAPs are arranged as one directory per class:

```text
dataset/
  amazon/*.pcap
  netflix/*.pcap
  ...
```

```bash
python scripts/preprocess.py \
  --data-root /path/to/dataset \
  --output-dir data/processed_dataset \
  --max-packets 30 \
  --workers 4
```

### 2. Train censors and local surrogates

The split is class-stratified: 50% defender, 30% attacker, and 20% test.

```bash
python scripts/train_classifiers.py \
  --data-dir data/processed_dataset \
  --output-dir models/censor \
  --max-packets 30 \
  --epochs 20
```

### 3. Train NetGuise

Full-window training:

```bash
python scripts/train_morpher.py \
  --surrogate models/censor/Attacker_UniNet.pt \
  --interface full \
  --epochs 20
```

Masked-causal training is available with `--interface online`. `mixed` samples
between full-window and causal batches.

### 4. Generate a plan

Full-window mode:

```bash
python scripts/morph.py \
  --input data/raw_data/input.pcap \
  --target-class youtube \
  --mode full \
  --plan results/full-window.json
```

Masked-causal online mode:

```bash
python scripts/morph.py \
  --input data/raw_data/input.pcap \
  --target-class youtube \
  --mode online \
  --plan results/online.json \
  --output-pcap results/online-timing.pcap
```

The optional PCAP output materializes non-negative delays. Padding remains in
the JSON plan because inserting bytes into an arbitrary TCP payload would not
preserve application semantics. A deployment endpoint must apply padding in
the tunnel encapsulation layer assumed by the threat model.

### 5. Validate constraints

```bash
python scripts/validate.py --plan results/online.json
```

The validator checks packet count, direction, additive-only actions, MTU, and
the configured bandwidth and latency budgets. It exits with a non-zero status
when any check fails.

## Online API

`netguise.online.CausalMorpherSession` exposes the packet-at-a-time interface.
Each call to `observe()` accepts one `[size, direction, IAT]` vector and returns
the padding and delay for that packet. The session contains no deployed-censor
query path.

## Checkpoints

The revised models use self-describing `.pt` checkpoints containing the model
type, architecture configuration, class names, normalization statistics, and
weights. Existing `.pth` files are retained as legacy outputs and are not
silently loaded into the revised architectures. See `models/README.md`.

## Scope and safety

The implementation is intended for controlled robustness evaluation on traffic
owned by the experimenter or used with explicit authorization. The online mode
implements the manuscript's masked-causal interface; it is not presented as a
general-purpose censorship bypass deployment.

## License

MIT. See `LICENSE`.
