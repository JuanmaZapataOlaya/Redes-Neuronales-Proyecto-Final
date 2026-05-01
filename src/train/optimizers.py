"""Fabricas de optimizadores."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn, optim


def build_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    """Crea Adam usando solo parametros entrenables."""

    trainable_parameters: Iterable[torch.nn.Parameter] = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return optim.Adam(trainable_parameters, lr=learning_rate)
