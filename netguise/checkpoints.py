from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from .models import FastFlowClassifier, NetGuiseMorpher, UniNetClassifier


FORMAT_VERSION = 1


def build_model(
    model_name: str,
    config: Dict[str, Any],
    feature_mean: Optional[Iterable[float]] = None,
    feature_scale: Optional[Iterable[float]] = None,
):
    common = {"feature_mean": feature_mean, "feature_scale": feature_scale}
    if model_name == "uninet":
        return UniNetClassifier(**config, **common)
    if model_name == "fastflow":
        return FastFlowClassifier(**config, **common)
    if model_name == "netguise":
        return NetGuiseMorpher(**config, **common)
    raise ValueError(f"unknown model type: {model_name}")


def save_checkpoint(
    path: str | Path,
    model_name: str,
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    classes: Iterable[str],
    feature_mean: Iterable[float],
    feature_scale: Iterable[float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_config": model_config,
        "classes": [str(value) for value in classes],
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "state_dict": model.state_dict(),
        "extra": {} if extra is None else extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(
            "legacy state_dict checkpoint detected; retrain with scripts/train_classifiers.py "
            "or scripts/train_morpher.py to create a self-describing checkpoint"
        )
    model = build_model(
        payload["model_name"],
        payload["model_config"],
        payload.get("feature_mean"),
        payload.get("feature_scale"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
