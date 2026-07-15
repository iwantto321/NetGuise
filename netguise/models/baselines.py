from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import nn

from .common import FeatureNormalizer, sequence_lengths, take_last_valid


class UniNetClassifier(nn.Module):
    """Packet/flow adaptation of UniNet's T-Attent classifier.

    The early-flow dataset exposes packet metadata rather than the full
    session-level T-Matrix. A learned flow-summary token supplies the available
    coarse granularity, while packet tokens retain the fine-grained sequence.
    """

    def __init__(
        self,
        num_classes: int,
        max_packets: int = 30,
        d_model: int = 64,
        num_heads: int = 4,
        feedforward_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_mean: Optional[Iterable[float]] = None,
        feature_scale: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_packets = max_packets
        self.normalizer = FeatureNormalizer(feature_mean, feature_scale)
        self.packet_embedding = nn.Linear(3, d_model)
        self.flow_embedding = nn.Linear(7, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, max_packets + 1, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _flow_summary(self, x: torch.Tensor) -> torch.Tensor:
        valid = x[..., 0].gt(0)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mask = valid.unsqueeze(-1)
        normalized = self.normalizer(x)
        mean = (normalized * mask).sum(dim=1) / count
        centered = (normalized - mean.unsqueeze(1)) * mask
        std = torch.sqrt(centered.square().sum(dim=1) / count + 1e-6)
        fraction = count / x.shape[1]
        return torch.cat([mean, std, fraction], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_packets:
            x = x[:, : self.max_packets]
        padding = x[..., 0].le(0)
        packet_tokens = self.packet_embedding(self.normalizer(x))
        flow_token = self.flow_embedding(self._flow_summary(x)).unsqueeze(1)
        tokens = torch.cat([flow_token, packet_tokens], dim=1)

        segment_ids = torch.zeros(
            (x.shape[0], x.shape[1] + 1), dtype=torch.long, device=x.device
        )
        segment_ids[:, 0] = 1
        tokens = (
            tokens
            + self.segment_embedding(segment_ids)
            + self.position_embedding[:, : tokens.shape[1]]
        )
        key_padding = torch.cat(
            [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), padding],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding)
        return self.classifier(encoded[:, 0])


@dataclass(frozen=True)
class EarlyDecision:
    label: torch.Tensor
    confidence: torch.Tensor
    packets_used: torch.Tensor
    is_unknown: torch.Tensor


class FastFlowClassifier(nn.Module):
    """Dual-grained early-flow classifier inspired by FastFlow.

    One LSTM processes the packet sequence and a second LSTM processes
    time-interval aggregates. ``predict_early`` provides the confidence-based
    dynamic stopping interface used in online evaluation.
    """

    def __init__(
        self,
        num_classes: int,
        max_packets: int = 30,
        hidden_size: int = 64,
        slot_width: float = 0.10,
        dropout: float = 0.1,
        feature_mean: Optional[Iterable[float]] = None,
        feature_scale: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_packets = max_packets
        self.slot_width = slot_width
        self.normalizer = FeatureNormalizer(feature_mean, feature_scale)
        self.packet_lstm = nn.LSTM(3, hidden_size, batch_first=True)
        self.slot_lstm = nn.LSTM(4, hidden_size, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def _time_slots(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = x[..., 0].gt(0)
        elapsed = torch.cumsum(x[..., 2].clamp_min(0), dim=1)
        slot_index = torch.floor(elapsed / self.slot_width).long()
        slot_index = slot_index.clamp(min=0, max=self.max_packets - 1)

        slot_input = torch.stack(
            [
                valid.float(),
                x[..., 0] / 1500.0,
                (x[..., 0] * x[..., 1]) / 1500.0,
                x[..., 2] / self.slot_width,
            ],
            dim=-1,
        )
        slot_input = slot_input * valid.unsqueeze(-1)
        slots = x.new_zeros((x.shape[0], self.max_packets, 4))
        slots.scatter_add_(
            1, slot_index.unsqueeze(-1).expand(-1, -1, 4), slot_input
        )
        slot_lengths = slots[..., 0].gt(0).sum(dim=1).clamp_min(1)
        return slots, slot_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_packets:
            x = x[:, : self.max_packets]
        packet_lengths = sequence_lengths(x)
        packet_output, _ = self.packet_lstm(self.normalizer(x))
        packet_state = take_last_valid(packet_output, packet_lengths)

        slots, slot_lengths = self._time_slots(x)
        slot_output, _ = self.slot_lstm(slots)
        slot_state = take_last_valid(slot_output, slot_lengths)

        gate = self.fusion_gate(torch.cat([packet_state, slot_state], dim=-1))
        fused = gate * packet_state + (1.0 - gate) * slot_state
        return self.classifier(fused)

    @torch.no_grad()
    def predict_early(
        self,
        x: torch.Tensor,
        decision_points: Sequence[int] = (3, 5, 10, 15, 20, 30),
        confidence_threshold: float = 0.90,
        unknown_threshold: float = 0.50,
    ) -> EarlyDecision:
        batch = x.shape[0]
        chosen_label = torch.zeros(batch, dtype=torch.long, device=x.device)
        chosen_confidence = torch.zeros(batch, device=x.device)
        packets_used = torch.full(
            (batch,), min(x.shape[1], self.max_packets), dtype=torch.long, device=x.device
        )
        decided = torch.zeros(batch, dtype=torch.bool, device=x.device)

        usable_points = sorted({min(int(k), x.shape[1]) for k in decision_points if k > 0})
        for point in usable_points:
            prefix = x.clone()
            prefix[:, point:] = 0
            probabilities = self(prefix).softmax(dim=-1)
            confidence, label = probabilities.max(dim=-1)
            take = (~decided) & (confidence >= confidence_threshold)
            chosen_label[take] = label[take]
            chosen_confidence[take] = confidence[take]
            packets_used[take] = point
            decided |= take

        final_probabilities = self(x).softmax(dim=-1)
        final_confidence, final_label = final_probabilities.max(dim=-1)
        chosen_label[~decided] = final_label[~decided]
        chosen_confidence[~decided] = final_confidence[~decided]
        return EarlyDecision(
            label=chosen_label,
            confidence=chosen_confidence,
            packets_used=packets_used,
            is_unknown=chosen_confidence < unknown_threshold,
        )
