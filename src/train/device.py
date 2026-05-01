"""Seleccion de dispositivo de computo."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Usa GPU si CUDA esta disponible; en caso contrario usa CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
