# Artifact guide

This guide separates quick correctness checks from full model training.

## Quick verification

After installing the dependencies, run:

```bash
pytest
```

The tests cover the following implementation claims:

| Claim | Check |
|---|---|
| Sigmoid action projection | Action fractions are bounded and projected into remaining budgets |
| Additive-only morphing | Packet size and IAT never decrease |
| Online causality | Changing future source packets cannot change already committed actions |
| Commit-once execution | The causal interface returns exactly one action per observed packet |
| Direction and count preservation | The feature-level validator compares source and morphed windows |
| FastFlow adaptation | Packet and time-slot branches produce early decisions |
| UniNet adaptation | Multi-granular tokens accept padded 30-packet inputs |

## Full workflow

1. Run `scripts/train_classifiers.py` to create defender and attacker models.
2. Run `scripts/train_morpher.py` with the desired observation interface.
3. Run `scripts/morph.py` on an owned or authorized PCAP.
4. Run `scripts/validate.py` on the generated JSON plan.

Default hyperparameters are stored in `configs/default.json` and match the
manuscript configuration: 30 packets, three packet features, 1500-byte MTU,
30% bandwidth budget, 50% latency budget, two encoder layers, 64-dimensional
embeddings, 128-dimensional feed-forward layers, four attention heads, batch
size 128, 20 epochs, AdamW, cosine learning-rate decay, and a learning rate of
5e-4.

## Expected outputs

- `data/splits/split_indices.npz`: reproducible defender/attacker/test split.
- `models/censor/*.pt`: self-describing neural classifier checkpoints.
- `models/censor/*.joblib`: classical baseline models.
- `models/morpher/netguise.pt`: revised Sigmoid-projection morpher.
- `results/*.json`: feature-level morphing plans and metadata.
- `results/*.pcap`: optional timing-shaped traces; padding is not inserted into
  application payloads.

## Limitations

- The included early-flow UniNet adaptation uses packet and flow granularity;
  the input dataset does not provide the full session-level T-Matrix.
- The FastFlow adaptation implements dual-grained LSTM classification and
  confidence stopping. It does not claim to reproduce details absent from the
  supplied short paper.
- A JSON plan records tunnel-layer padding. Materializing that padding requires
  the controlled two-ended tunnel endpoint; the offline PCAP writer changes
  timing only.
- End-to-end replay tools such as Mininet, Ryu, Wireshark, and tcpreplay are
  environment-specific and are not replaced by fabricated validation output.
