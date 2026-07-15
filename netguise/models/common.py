from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class FeatureNormalizer(nn.Module):
    """Standardize packet features while retaining zero-padding."""

    def __init__(
        self,
        mean: Optional[Iterable[float]] = None,
        scale: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(
            [0.0, 0.0, 0.0] if mean is None else list(mean), dtype=torch.float32
        )
        scale_tensor = torch.as_tensor(
            [1.0, 1.0, 1.0] if scale is None else list(scale), dtype=torch.float32
        ).clamp_min(1e-8)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("scale", scale_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = x[..., 0].gt(0).unsqueeze(-1)
        normalized = (x - self.mean) / self.scale
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.mean


def sequence_lengths(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0].gt(0).sum(dim=1).clamp_min(1)


def take_last_valid(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence.shape[-1])
    return sequence.gather(1, index).squeeze(1)
