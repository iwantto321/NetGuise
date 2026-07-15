from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass(frozen=True)
class DatasetSplit:
    defender: np.ndarray
    attacker: np.ndarray
    test: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(labels: np.ndarray, seed: int = 42) -> DatasetSplit:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    remaining, test = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=seed
    )
    defender, attacker = train_test_split(
        remaining,
        train_size=0.625,
        stratify=labels[remaining],
        random_state=seed,
    )
    return DatasetSplit(defender=defender, attacker=attacker, test=test)


def save_split(path: str | Path, split: DatasetSplit) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        defender=split.defender,
        attacker=split.attacker,
        test=split.test,
    )


def load_split(path: str | Path) -> DatasetSplit:
    values = np.load(path)
    return DatasetSplit(values["defender"], values["attacker"], values["test"])


def _validation_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    correct = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += int((model(x).argmax(dim=1) == y).sum())
            count += len(y)
    return correct / max(count, 1)


def train_neural_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-5,
    decision_points: Sequence[int] = (5, 10, 15, 20, 30),
    patience: int = 5,
) -> tuple[nn.Module, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            point = min(random.choice(tuple(decision_points)), x.shape[1])
            prefix = x.clone()
            prefix[:, point:] = 0
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * criterion(model(x), y) + 0.5 * criterion(model(prefix), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        accuracy = _validation_accuracy(model, validation_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_accuracy
