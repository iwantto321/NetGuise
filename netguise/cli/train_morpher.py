from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from netguise.checkpoints import load_checkpoint, save_checkpoint
from netguise.config import ExperimentConfig
from netguise.data import load_processed
from netguise.models import NetGuiseMorpher
from netguise.reference import ReferencePool
from netguise.training import load_split, set_seed, split_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the NetGuise morpher")
    parser.add_argument("--data-dir", "--data_dir", default="data/processed_dataset")
    parser.add_argument("--split-file", default="data/splits/split_indices.npz")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--surrogate", default="models/censor/Attacker_UniNet.pt")
    parser.add_argument("--output", default="models/morpher/netguise.pt")
    parser.add_argument("--interface", choices=("full", "online", "mixed"), default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--cost-weight", type=float, default=0.01)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    defaults = ExperimentConfig.load(args.config)
    args.epochs = defaults.training.epochs if args.epochs is None else args.epochs
    args.batch_size = (
        defaults.training.batch_size if args.batch_size is None else args.batch_size
    )
    args.max_packets = (
        defaults.traffic.max_packets if args.max_packets is None else args.max_packets
    )
    args.learning_rate = (
        defaults.training.learning_rate
        if args.learning_rate is None
        else args.learning_rate
    )
    args.weight_decay = (
        defaults.training.weight_decay if args.weight_decay is None else args.weight_decay
    )
    args.seed = defaults.training.seed if args.seed is None else args.seed
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x, y, classes = load_processed(args.data_dir)
    x = x[:, : args.max_packets]
    split_path = Path(args.split_file)
    split = load_split(split_path) if split_path.exists() else split_indices(y, args.seed)
    train_x = x[split.attacker]
    train_y = y[split.attacker]
    reference_pool = ReferencePool(train_x, train_y)
    surrogate, surrogate_payload = load_checkpoint(args.surrogate, device=device)
    if surrogate_payload["model_name"] not in {"fastflow", "uninet"}:
        raise SystemExit("the surrogate must be a differentiable FastFlow or UniNet model")
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)

    config = {
        "max_packets": x.shape[1],
        "d_model": defaults.morpher.d_model,
        "num_heads": defaults.morpher.attention_heads,
        "feedforward_dim": defaults.morpher.feedforward_dim,
        "num_layers": defaults.morpher.encoder_layers,
        "dropout": defaults.morpher.dropout,
        "mtu": defaults.traffic.mtu,
        "bandwidth_ratio": defaults.traffic.bandwidth_ratio,
        "latency_ratio": defaults.traffic.latency_ratio,
    }
    mean = surrogate_payload["feature_mean"]
    scale = surrogate_payload["feature_scale"]
    morpher = NetGuiseMorpher(
        **config, feature_mean=mean, feature_scale=scale
    ).to(device)
    optimizer = torch.optim.AdamW(
        morpher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_x))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    decision_points = (5, 10, 15, 20, 30)

    for epoch in range(args.epochs):
        morpher.train()
        epoch_loss = 0.0
        samples = 0
        for (indices,) in loader:
            index_values = indices.numpy()
            source_np = train_x[index_values]
            interface = args.interface
            if interface == "mixed":
                interface = "online" if random.random() < 0.5 else "full"
            target_np, target_label_np = reference_pool.select_batch(
                source_np,
                train_y[index_values],
                candidate_count=args.candidate_count,
                rng=rng,
                observed_packets=1 if interface == "online" else None,
            )
            source = torch.as_tensor(source_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            target_labels = torch.as_tensor(
                target_label_np, dtype=torch.long, device=device
            )
            if interface == "online":
                point = min(random.choice(decision_points), source.shape[1])
                output = morpher.forward_causal(source, target, decision_point=point)
                classifier_input = output.morphed.clone()
                classifier_input[:, point:] = 0
            else:
                output = morpher(source, target)
                classifier_input = output.morphed

            logits = surrogate(classifier_input)
            adversarial_loss = F.cross_entropy(logits, target_labels)
            valid = source[..., 0].gt(0)
            source_bytes = source[..., 0].sum(dim=1).clamp_min(1e-8)
            source_time = source[..., 2].sum(dim=1).clamp_min(1e-8)
            bandwidth = (output.actions[..., 0] * valid).sum(dim=1) / source_bytes
            latency = (output.actions[..., 1] * valid).sum(dim=1) / source_time
            cost = (bandwidth + latency).mean()
            loss = adversarial_loss + args.cost_weight * cost
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
            samples += len(indices)
        scheduler.step()
        logging.info("epoch %d loss %.5f", epoch + 1, epoch_loss / max(samples, 1))

    save_checkpoint(
        args.output,
        "netguise",
        morpher,
        config,
        classes.astype(str),
        mean,
        scale,
        extra={
            "training_interface": args.interface,
            "surrogate": str(args.surrogate),
            "candidate_count": args.candidate_count,
        },
    )
    logging.info("saved morpher to %s", args.output)


if __name__ == "__main__":
    main()
