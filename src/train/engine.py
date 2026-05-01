"""Funciones puras del ciclo de entrenamiento y evaluacion."""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entrena una epoca y retorna perdida y exactitud."""

    model.train()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    return epoch_loss, epoch_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evalua sin gradientes y retorna perdida, exactitud y predicciones."""

    model.eval()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    return epoch_loss, epoch_accuracy, all_predictions, all_targets
