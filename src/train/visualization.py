"""Visualizacion de metricas de entrenamiento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> None:
    """Guarda curvas de perdida y exactitud."""

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Exactitud")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
