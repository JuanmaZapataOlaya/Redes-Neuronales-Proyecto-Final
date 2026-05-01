"""Punto de entrada CLI para entrenar modelos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn

from src.data import DataLoaderConfig, create_dataloaders
from src.data.constants import CLASS_TO_INDEX
from src.models import BaseCNN
from src.train.device import get_device
from src.train.engine import evaluate, train_one_epoch
from src.train.optimizers import build_optimizer
from src.train.visualization import save_training_plot


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI para experimentos reproducibles."""

    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo BaseCNN para deteccion de neumonia."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Ruta al dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano del batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Tasa de aprendizaje.")
    parser.add_argument("--image-size", type=int, default=224, help="Resolucion de entrada.")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers del DataLoader.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directorio donde se guardan pesos y graficas.",
    )
    return parser.parse_args()


def main() -> None:
    """Orquesta carga de datos, modelo, entrenamiento y evaluacion."""

    args = parse_args()
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_dataloaders(
        DataLoaderConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            pin_memory=device.type == "cuda",
        )
    )

    model = BaseCNN(num_classes=len(CLASS_TO_INDEX)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, learning_rate=args.learning_rate)

    best_val_accuracy = -1.0
    best_model_path = args.output_dir / "best_model.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Dispositivo: {device}")
    print("Modelo seleccionado: BaseCNN")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, _, _ = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoca {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "model_name": "base",
                    "model_state_dict": model.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "image_size": args.image_size,
                    "class_to_index": CLASS_TO_INDEX,
                },
                best_model_path,
            )
            print(f"Nuevo mejor modelo guardado en: {best_model_path}")

    save_training_plot(history, args.output_dir / "training_curves.png")

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if "test" in dataloaders:
        test_loss, test_acc, predictions, targets = evaluate(
            model=model,
            dataloader=dataloaders["test"],
            criterion=criterion,
            device=device,
        )
        print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}")
        print(
            classification_report(
                targets,
                predictions,
                target_names=["NORMAL", "PNEUMONIA"],
                digits=4,
            )
        )

    print(f"Mejor exactitud de validacion: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
