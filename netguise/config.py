from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrafficConfig:
    max_packets: int = 30
    feature_dim: int = 3
    mtu: float = 1500.0
    bandwidth_ratio: float = 0.30
    latency_ratio: float = 0.50


@dataclass
class MorpherConfig:
    d_model: int = 64
    feedforward_dim: int = 128
    encoder_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    seed: int = 42


@dataclass
class ExperimentConfig:
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    morpher: MorpherConfig = field(default_factory=MorpherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            traffic=TrafficConfig(**values.get("traffic", {})),
            morpher=MorpherConfig(**values.get("morpher", {})),
            training=TrainingConfig(**values.get("training", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
