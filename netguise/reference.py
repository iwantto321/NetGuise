from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CompatibilityWeights:
    size: float = 1.0 / 3.0
    timing: float = 1.0 / 3.0
    direction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if not np.isclose(self.size + self.timing + self.direction, 1.0):
            raise ValueError("compatibility weights must sum to one")


def morphability_score(
    source: np.ndarray,
    target: np.ndarray,
    bandwidth_ratio: float = 0.30,
    latency_ratio: float = 0.50,
    weights: CompatibilityWeights = CompatibilityWeights(),
    epsilon: float = 1e-8,
) -> float:
    """Compute the paper's size, timing, and direction compatibility score."""

    valid = source[:, 0] > 0
    if not np.any(valid):
        return float("-inf")
    src = source[valid]
    tgt = target[: len(src)]

    size_budget = bandwidth_ratio * float(src[:, 0].sum())
    shrink_size = np.maximum(src[:, 0] - tgt[:, 0], 0.0).sum()
    add_size = np.maximum(tgt[:, 0] - src[:, 0], 0.0).sum()
    size_penalty = shrink_size + max(add_size - size_budget, 0.0)
    size_compatibility = np.exp(-size_penalty / (src[:, 0].sum() + epsilon))

    latency_budget = latency_ratio * float(src[:, 2].sum())
    advance_time = np.maximum(src[:, 2] - tgt[:, 2], 0.0).sum()
    add_time = np.maximum(tgt[:, 2] - src[:, 2], 0.0).sum()
    timing_penalty = advance_time + max(add_time - latency_budget, 0.0)
    timing_compatibility = np.exp(-timing_penalty / (src[:, 2].sum() + epsilon))

    target_valid = tgt[:, 0] > 0
    comparable = target_valid & valid[: len(tgt)]
    if np.any(comparable):
        direction_compatibility = np.mean(
            np.sign(src[comparable, 1]) == np.sign(tgt[comparable, 1])
        )
    else:
        direction_compatibility = 0.0

    return float(
        weights.size * size_compatibility
        + weights.timing * timing_compatibility
        + weights.direction * direction_compatibility
    )


class ReferencePool:
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError("reference features and labels must have equal length")
        self.features = features
        self.labels = labels
        self.by_class = {
            int(label): np.flatnonzero(labels == label)
            for label in np.unique(labels)
        }

    def select(
        self,
        source: np.ndarray,
        target_class: int,
        candidate_count: int = 32,
        seed: Optional[int] = None,
        observed_packets: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        candidates = self.by_class.get(int(target_class))
        if candidates is None or len(candidates) == 0:
            raise ValueError(f"reference pool has no samples for class {target_class}")
        rng = np.random.default_rng(seed)
        count = min(candidate_count, len(candidates))
        selected = rng.choice(candidates, size=count, replace=False)
        scoring_source = source.copy()
        if observed_packets is not None:
            scoring_source[observed_packets:] = 0
        scores = [
            morphability_score(scoring_source, self.features[index])
            for index in selected
        ]
        position = int(np.argmax(scores))
        index = int(selected[position])
        return self.features[index].copy(), index, float(scores[position])

    def select_batch(
        self,
        sources: np.ndarray,
        source_labels: np.ndarray,
        candidate_count: int = 32,
        rng: Optional[np.random.Generator] = None,
        observed_packets: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng() if rng is None else rng
        available = np.asarray(sorted(self.by_class), dtype=np.int64)
        targets = []
        target_labels = []
        for source, source_label in zip(sources, source_labels):
            choices = available[available != int(source_label)]
            if len(choices) == 0:
                raise ValueError("target selection requires at least two classes")
            target_class = int(rng.choice(choices))
            target, _, _ = self.select(
                source,
                target_class,
                candidate_count=candidate_count,
                seed=int(rng.integers(0, 2**32 - 1)),
                observed_packets=observed_packets,
            )
            targets.append(target)
            target_labels.append(target_class)
        return np.asarray(targets, dtype=np.float32), np.asarray(target_labels, dtype=np.int64)
