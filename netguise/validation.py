from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationReport:
    packet_count_preserved: bool
    direction_preserved: bool
    additive_only: bool
    mtu_respected: bool
    bandwidth_budget_respected: bool
    latency_budget_respected: bool
    bandwidth_overhead: float
    latency_overhead: float

    @property
    def passed(self) -> bool:
        return all(
            (
                self.packet_count_preserved,
                self.direction_preserved,
                self.additive_only,
                self.mtu_respected,
                self.bandwidth_budget_respected,
                self.latency_budget_respected,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["passed"] = self.passed
        return values


def validate_features(
    source: np.ndarray,
    morphed: np.ndarray,
    mtu: float = 1500.0,
    bandwidth_ratio: float = 0.30,
    latency_ratio: float = 0.50,
    tolerance: float = 1e-5,
) -> ValidationReport:
    source_valid = source[:, 0] > 0
    morphed_valid = morphed[:, 0] > 0
    size_delta = morphed[:, 0] - source[:, 0]
    time_delta = morphed[:, 2] - source[:, 2]
    source_size = max(float(source[source_valid, 0].sum()), tolerance)
    source_time = max(float(source[source_valid, 2].sum()), tolerance)
    bandwidth_overhead = float(size_delta[source_valid].sum() / source_size)
    latency_overhead = float(time_delta[source_valid].sum() / source_time)
    return ValidationReport(
        packet_count_preserved=bool(np.array_equal(source_valid, morphed_valid)),
        direction_preserved=bool(
            np.allclose(source[source_valid, 1], morphed[source_valid, 1], atol=tolerance)
        ),
        additive_only=bool(
            np.all(size_delta[source_valid] >= -tolerance)
            and np.all(time_delta[source_valid] >= -tolerance)
        ),
        mtu_respected=bool(np.all(morphed[source_valid, 0] <= mtu + tolerance)),
        bandwidth_budget_respected=bandwidth_overhead <= bandwidth_ratio + tolerance,
        latency_budget_respected=latency_overhead <= latency_ratio + tolerance,
        bandwidth_overhead=bandwidth_overhead,
        latency_overhead=latency_overhead,
    )
