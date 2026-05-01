# Diagnostico de Neumonia con Redes Neuronales

Proyecto academico en Python y PyTorch para clasificar radiografias de torax como `NORMAL` o `PNEUMONIA`. La estructura se mantiene simple: un README y notebooks autocontenidos para comparar tres arquitecturas.

> Este proyecto es educativo. No debe utilizarse como herramienta clinica sin validacion medica, auditoria de sesgos y evaluacion regulatoria.

## Estructura

```text
.
|-- README.md
|-- arquitectura_base_neumonia.ipynb
|-- arquitectura_cnn_avanzada_neumonia.ipynb
|-- arquitectura_transfer_learning_neumonia.ipynb
`-- .gitignore
```

Cada notebook contiene todo el flujo:

- Configuracion reproducible.
- Carga del dataset desde carpetas.
- Transformaciones de entrenamiento y validacion.
- Arquitectura del modelo correspondiente.
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

Si tienes GPU NVIDIA, instala la version de PyTorch compatible con tu CUDA desde la guia oficial de PyTorch. Los notebooks detectan CUDA automaticamente cuando esta disponible.

## Uso

Descarga el dataset en Kaggle:

https://www.kaggle.com/datasets/assemelqirsh/chest-x-ray-dataset?resource=download

Para este caso solo vamos a usar `chest_xray`, respetando la estructura con la que ya viene:

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

Cambia el nombre de la carpeta a `data`; si no, fallara la celda de carga de datos.

Abre el notebook que quieras entrenar:

```bash
jupyter notebook arquitectura_base_neumonia.ipynb
jupyter notebook arquitectura_cnn_avanzada_neumonia.ipynb
jupyter notebook arquitectura_transfer_learning_neumonia.ipynb
```

Ejecuta las celdas de arriba hacia abajo. Antes de entrenar puedes ajustar estos parametros en la celda de configuracion:

- `DATA_DIR`: ruta del dataset.
- `OUTPUT_DIR`: carpeta para pesos y graficas.
- `IMAGE_SIZE`: resolucion de entrada.
- `BATCH_SIZE`: tamano del lote.
- `EPOCHS`: numero de epocas.
- `LEARNING_RATE`: tasa de aprendizaje.

Si ejecutas en Kaggle, corre la celda opcional marcada como `CELDA OPCIONAL PARA KAGGLE` despues de imports/configuracion. Esa celda busca el dataset dentro de `/kaggle/input` y ajusta `DATA_DIR` automaticamente.

## Arquitectura base

`BaseCNN` usa tres bloques convolucionales:

```text
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
AdaptiveAvgPool2d -> Linear -> ReLU -> Linear
```

## CNN avanzada

`AdvancedCNN` usa bloques con dos convoluciones, `BatchNorm2d`, `ReLU`, `MaxPool2d` y `Dropout`. Es una segunda prueba mas robusta para reducir sobreajuste y comparar contra la arquitectura base.

## Transfer learning

El notebook de transferencia usa `ResNet-50` de `torchvision`, reemplaza la capa final por una salida de dos clases y permite congelar el backbone con `FREEZE_BACKBONE = True`.

## Salidas

Durante el entrenamiento se crea la carpeta `models/` con:

```text
models/
  best_base_cnn.pth
  best_advanced_cnn.pth
  best_resnet50_transfer.pth
  training_curves.png
  training_curves_advanced_cnn.png
  training_curves_resnet50_transfer.png
```

Cada checkpoint guarda los pesos del mejor modelo segun exactitud de validacion, el tamano de imagen y el mapeo de clases.

## Siguiente paso sugerido

Entrena los tres notebooks con los mismos splits y compara exactitud, precision, recall y F1-score sobre `test`.

# Resultados Obtenidos

## Arquitectura Base
<img width="500" alt="image" src="https://github.com/user-attachments/assets/3adfada4-566f-4a04-a626-dc520a144eaa" />

<img width="500"  alt="image" src="https://github.com/user-attachments/assets/a69fc5c6-f2ff-4a74-9403-ea4ecde3ce73" />

