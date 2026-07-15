from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from .common import FeatureNormalizer


@dataclass
class MorpherOutput:
    morphed: torch.Tensor
    actions: torch.Tensor
    action_logits: torch.Tensor


class SharedPacketEncoder(nn.Module):
    def __init__(
        self,
        max_packets: int,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        feature_mean: Optional[Iterable[float]],
        feature_scale: Optional[Iterable[float]],
    ) -> None:
        super().__init__()
        self.max_packets = max_packets
        self.normalizer = FeatureNormalizer(feature_mean, feature_scale)
        self.input_projection = nn.Linear(3, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, max_packets, d_model))
        self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
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

    def forward(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        keep_mask_tokens: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x[:, : self.max_packets]
        tokens = self.input_projection(self.normalizer(x))
        padding = x[..., 0].le(0)
        if observed_mask is not None:
            observed_mask = observed_mask[:, : x.shape[1]]
            hidden = ~observed_mask
            tokens = torch.where(hidden.unsqueeze(-1), self.mask_token, tokens)
            if keep_mask_tokens:
                padding = torch.zeros_like(padding)
            else:
                padding = padding & observed_mask
        tokens = tokens + self.position_embedding[:, : x.shape[1]]
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return encoded, padding


class NetGuiseMorpher(nn.Module):
    """Target-conditioned morpher with Sigmoid budget projection."""

    def __init__(
        self,
        max_packets: int = 30,
        d_model: int = 64,
        num_heads: int = 4,
        feedforward_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        mtu: float = 1500.0,
        bandwidth_ratio: float = 0.30,
        latency_ratio: float = 0.50,
        feature_mean: Optional[Iterable[float]] = None,
        feature_scale: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.max_packets = max_packets
        self.mtu = float(mtu)
        self.bandwidth_ratio = float(bandwidth_ratio)
        self.latency_ratio = float(latency_ratio)
        self.encoder = SharedPacketEncoder(
            max_packets=max_packets,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            dropout=dropout,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        self.source_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.alignment_norm = nn.LayerNorm(d_model)
        self.alignment_mlp = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, 2),
        )

    def action_logits(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_hidden, _ = self.encoder(
            source,
            observed_mask=observed_mask,
            keep_mask_tokens=observed_mask is not None,
        )
        target_hidden, target_padding = self.encoder(target)
        aligned, _ = self.cross_attention(
            query=self.source_norm(source_hidden),
            key=self.target_norm(target_hidden),
            value=self.target_norm(target_hidden),
            key_padding_mask=target_padding,
            need_weights=False,
        )
        hidden = source_hidden + aligned
        hidden = hidden + self.alignment_mlp(self.alignment_norm(hidden))
        return self.decoder(hidden)

    def _compose(self, source: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                source[..., 0] + actions[..., 0],
                source[..., 1],
                source[..., 2] + actions[..., 1],
            ],
            dim=-1,
        )

    def project_full(
        self, source: torch.Tensor, action_logits: torch.Tensor
    ) -> torch.Tensor:
        valid = source[..., 0].gt(0)
        size_budget = source[..., 0].sum(dim=1) * self.bandwidth_ratio
        latency_budget = source[..., 2].sum(dim=1) * self.latency_ratio
        remaining_size = size_budget
        remaining_latency = latency_budget
        actions = []

        fractions = torch.sigmoid(action_logits)
        for index in range(source.shape[1]):
            is_valid = valid[:, index].float()
            mtu_room = (self.mtu - source[:, index, 0]).clamp_min(0)
            padding_cap = torch.minimum(mtu_room, remaining_size.clamp_min(0))
            padding = fractions[:, index, 0] * padding_cap * is_valid
            delay = (
                fractions[:, index, 1]
                * remaining_latency.clamp_min(0)
                * is_valid
            )
            remaining_size = remaining_size - padding
            remaining_latency = remaining_latency - delay
            actions.append(torch.stack([padding, delay], dim=-1))
        return torch.stack(actions, dim=1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> MorpherOutput:
        logits = self.action_logits(source, target)
        actions = self.project_full(source, logits)
        return MorpherOutput(self._compose(source, actions), actions, logits)

    def forward_causal(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        decision_point: Optional[int] = None,
    ) -> MorpherOutput:
        """Run the masked-causal, commit-once interface.

        Budgets become available with each observed packet. This prevents the
        projection layer from leaking future packet sizes or timing into the
        current action.
        """

        horizon = source.shape[1]
        if decision_point is not None:
            horizon = min(horizon, int(decision_point))
        remaining_size = source.new_zeros(source.shape[0])
        remaining_latency = source.new_zeros(source.shape[0])
        committed = []
        committed_logits = []

        for index in range(source.shape[1]):
            if index >= horizon:
                zero = source.new_zeros((source.shape[0], 2))
                committed.append(zero)
                committed_logits.append(zero)
                continue
            observed = torch.arange(source.shape[1], device=source.device)
            observed = observed.le(index).unsqueeze(0).expand(source.shape[0], -1)
            logits = self.action_logits(source, target, observed_mask=observed)[:, index]
            fractions = torch.sigmoid(logits)
            valid = source[:, index, 0].gt(0).float()

            remaining_size = (
                remaining_size
                + source[:, index, 0].clamp_min(0) * self.bandwidth_ratio
            )
            remaining_latency = (
                remaining_latency
                + source[:, index, 2].clamp_min(0) * self.latency_ratio
            )
            mtu_room = (self.mtu - source[:, index, 0]).clamp_min(0)
            padding_cap = torch.minimum(mtu_room, remaining_size.clamp_min(0))
            padding = fractions[:, 0] * padding_cap * valid
            delay = fractions[:, 1] * remaining_latency.clamp_min(0) * valid
            action = torch.stack([padding, delay], dim=-1)
            remaining_size = remaining_size - padding
            remaining_latency = remaining_latency - delay
            committed.append(action)
            committed_logits.append(logits)

        actions = torch.stack(committed, dim=1)
        logits = torch.stack(committed_logits, dim=1)
        return MorpherOutput(self._compose(source, actions), actions, logits)
