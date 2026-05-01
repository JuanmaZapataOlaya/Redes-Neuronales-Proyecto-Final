# Diagnostico de Neumonia con CNN Base

Proyecto academico en Python y PyTorch para clasificar radiografias de torax como `NORMAL` o `PNEUMONIA`. La estructura se simplifico a un README y un notebook autocontenido con la primera arquitectura base implementada de principio a fin.

> Este proyecto es educativo. No debe utilizarse como herramienta clinica sin validacion medica, auditoria de sesgos y evaluacion regulatoria.

## Estructura

```text
.
|-- README.md
|-- arquitectura_base_neumonia.ipynb
`-- .gitignore
```

El notebook contiene todo el flujo:

- Configuracion reproducible.
- Carga del dataset desde carpetas.
- Transformaciones de entrenamiento y validacion.
- Arquitectura `BaseCNN`.
- Ciclo de entrenamiento y evaluacion.
- Guardado del mejor modelo.
- Graficas de perdida y exactitud.
- Reporte final sobre `test` cuando ese split existe.

## Dataset esperado

Coloca las imagenes en una carpeta `data/` con esta organizacion:

```text
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

El split `test` es opcional. `train` y `val` son requeridos para entrenar.

## Dependencias

Instala las librerias necesarias en tu entorno:

```bash
pip install torch torchvision matplotlib scikit-learn pillow notebook
```

Si tienes GPU NVIDIA, instala la version de PyTorch compatible con tu CUDA desde la guia oficial de PyTorch. El notebook detecta CUDA automaticamente cuando esta disponible.

## Uso

Abre el notebook:

```bash
jupyter notebook arquitectura_base_neumonia.ipynb
```

Ejecuta las celdas de arriba hacia abajo. Antes de entrenar puedes ajustar estos parametros en la celda de configuracion:

- `DATA_DIR`: ruta del dataset.
- `OUTPUT_DIR`: carpeta para pesos y graficas.
- `IMAGE_SIZE`: resolucion de entrada.
- `BATCH_SIZE`: tamano del lote.
- `EPOCHS`: numero de epocas.
- `LEARNING_RATE`: tasa de aprendizaje.

## Arquitectura base

`BaseCNN` usa tres bloques convolucionales:

```text
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
AdaptiveAvgPool2d -> Linear -> ReLU -> Linear
```

La salida tiene dos clases: `NORMAL` y `PNEUMONIA`.

## Salidas

Durante el entrenamiento se crea la carpeta `models/` con:

```text
models/
  best_base_cnn.pth
  training_curves.png
```

El checkpoint guarda los pesos del mejor modelo segun exactitud de validacion, el tamano de imagen y el mapeo de clases.

## Siguiente paso sugerido

Una vez validada la CNN base, el siguiente paso natural es comparar contra una arquitectura con `BatchNorm2d` y `Dropout`, o contra transferencia de aprendizaje con una red pre-entrenada.
