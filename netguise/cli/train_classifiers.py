from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from netguise.checkpoints import save_checkpoint
from netguise.classical import train_ggfast, train_gradbp
from netguise.config import ExperimentConfig
from netguise.data import feature_statistics, load_processed
from netguise.models import FastFlowClassifier, UniNetClassifier
from netguise.training import (
    TrafficDataset,
    save_split,
    set_seed,
    split_indices,
    train_neural_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train censor and surrogate models")
    parser.add_argument("--data-dir", "--data_dir", default="data/processed_dataset")
    parser.add_argument("--output-dir", "--save_dir", default="models/censor")
    parser.add_argument("--split-file", default="data/splits/split_indices.npz")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--models", nargs="+", default=["ggfast", "gradbp", "fastflow", "uninet"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", "--batch_size", type=int)
    parser.add_argument("--max-packets", "--n_packets", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    return parser


def _neural_model(name: str, num_classes: int, max_packets: int, mean, scale):
    if name == "fastflow":
        config = {
            "num_classes": num_classes,
            "max_packets": max_packets,
            "hidden_size": 64,
            "slot_width": 0.10,
            "dropout": 0.1,
        }
        return FastFlowClassifier(**config, feature_mean=mean, feature_scale=scale), config
    if name == "uninet":
        config = {
            "num_classes": num_classes,
            "max_packets": max_packets,
            "d_model": 64,
            "num_heads": 4,
            "feedforward_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
        }
        return UniNetClassifier(**config, feature_mean=mean, feature_scale=scale), config
    raise ValueError(name)


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
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x, y, classes = load_processed(args.data_dir)
    x = x[:, : args.max_packets]
    split = split_indices(y, seed=args.seed)
    save_split(args.split_file, split)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for role, role_indices in (("Defender", split.defender), ("Attacker", split.attacker)):
        validation_count = max(
            len(np.unique(y[role_indices])), int(np.ceil(0.10 * len(role_indices)))
        )
        train_idx, validation_idx = train_test_split(
            role_indices,
            test_size=validation_count,
            stratify=y[role_indices],
            random_state=args.seed,
        )
        mean, scale = feature_statistics(x[train_idx])
        np.savez(output / f"{role.lower()}_normalization.npz", mean=mean, scale=scale)
        for name in args.models:
            name = name.lower()
            logging.info("training %s %s", role, name)
            if name == "ggfast":
                model = train_ggfast(x[train_idx], y[train_idx])
                accuracy = np.mean(model.predict(x[validation_idx]) == y[validation_idx])
                joblib.dump(model, output / f"{role}_GGFAST.joblib")
            elif name == "gradbp":
                model = train_gradbp(x[train_idx], y[train_idx])
                accuracy = np.mean(model.predict(x[validation_idx]) == y[validation_idx])
                joblib.dump(model, output / f"{role}_GradBP.joblib")
            elif name in {"fastflow", "uninet"}:
                model, config = _neural_model(
                    name, len(classes), x.shape[1], mean, scale
                )
                train_loader = DataLoader(
                    TrafficDataset(x[train_idx], y[train_idx]),
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                validation_loader = DataLoader(
                    TrafficDataset(x[validation_idx], y[validation_idx]),
                    batch_size=args.batch_size,
                )
                model, accuracy = train_neural_classifier(
                    model,
                    train_loader,
                    validation_loader,
                    device,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                display_name = {"fastflow": "FastFlow", "uninet": "UniNet"}[name]
                save_checkpoint(
                    output / f"{role}_{display_name}.pt",
                    name,
                    model,
                    config,
                    classes.astype(str),
                    mean,
                    scale,
                    extra={"validation_accuracy": float(accuracy), "role": role.lower()},
                )
            else:
                raise SystemExit(f"unknown model: {name}")
            logging.info("%s %s validation accuracy: %.4f", role, name, accuracy)

    np.save(output / "classes.npy", classes)
    np.save(output / "X_test.npy", x[split.test])
    np.save(output / "y_test.npy", y[split.test])


if __name__ == "__main__":
    main()
