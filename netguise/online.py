from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .models.morpher import NetGuiseMorpher


@dataclass(frozen=True)
class PacketAction:
    index: int
    padding_bytes: float
    delay_seconds: float


class CausalMorpherSession:
    """Stateful packet-at-a-time interface for masked-causal inference."""

    def __init__(
        self,
        model: NetGuiseMorpher,
        target_reference: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device if device is None else device
        target = target_reference.to(self.device, dtype=torch.float32)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        self.target = target
        self.source = torch.zeros(
            (1, model.max_packets, 3), dtype=torch.float32, device=self.device
        )
        self.position = 0
        self.remaining_size = torch.zeros(1, device=self.device)
        self.remaining_latency = torch.zeros(1, device=self.device)
        self.actions: list[PacketAction] = []

    @torch.no_grad()
    def observe(self, packet_features: torch.Tensor) -> PacketAction:
        if self.position >= self.model.max_packets:
            raise StopIteration("the early-stage window is complete")
        feature = packet_features.to(self.device, dtype=torch.float32).reshape(3)
        self.source[0, self.position] = feature
        observed = torch.arange(self.model.max_packets, device=self.device)
        observed = observed.le(self.position).unsqueeze(0)
        logits = self.model.action_logits(
            self.source, self.target, observed_mask=observed
        )[0, self.position]
        fractions = torch.sigmoid(logits)

        self.remaining_size += feature[0].clamp_min(0) * self.model.bandwidth_ratio
        self.remaining_latency += feature[2].clamp_min(0) * self.model.latency_ratio
        padding_cap = torch.minimum(
            (self.model.mtu - feature[0]).clamp_min(0), self.remaining_size
        )
        padding = fractions[0] * padding_cap
        delay = fractions[1] * self.remaining_latency
        if feature[0] <= 0:
            padding = padding.new_zeros(())
            delay = delay.new_zeros(())
        self.remaining_size -= padding
        self.remaining_latency -= delay

        action = PacketAction(
            index=self.position,
            padding_bytes=float(padding.cpu()),
            delay_seconds=float(delay.cpu()),
        )
        self.actions.append(action)
        self.position += 1
        return action
