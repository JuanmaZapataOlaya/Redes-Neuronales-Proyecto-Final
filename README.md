# Diagnostico de Neumonia con Redes Neuronales

Proyecto en Python 3.10+ y PyTorch para clasificar radiografias de torax como `NORMAL` o `PNEUMONIA` usando el dataset **Chest X-Ray Images Pneumonia**. La estructura esta pensada para trabajar en VS Code como un proyecto de software reproducible, modular y extensible.

## Problema a resolver

La neumonia puede producir patrones visibles en radiografias de torax, pero su interpretacion requiere criterio medico especializado. Este proyecto aborda el problema como una tarea de **clasificacion binaria de imagenes medicas**, donde una red neuronal aprende representaciones visuales para distinguir entre radiografias normales y radiografias compatibles con neumonia.

El objetivo academico es comparar tres enfoques de complejidad creciente:

- Una CNN base como linea de referencia.
- Una CNN avanzada con regularizacion y normalizacion.
- Un modelo con transferencia de aprendizaje usando ResNet-50 pre-entrenada.

> Importante: este proyecto es educativo y no debe utilizarse como herramienta clinica sin validacion medica, auditoria de sesgos y evaluacion regulatoria.

## Datos

El dataset **Chest X-Ray Images Pneumonia** contiene radiografias de torax organizadas en carpetas por clase. La version comun del dataset usa esta estructura:

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

Cada imagen se carga como RGB para mantener compatibilidad con arquitecturas pre-entrenadas de `torchvision`. Durante entrenamiento se aplican transformaciones basicas:

- Redimensionamiento a `224x224`.
- Normalizacion con medias y desviaciones estandar de ImageNet.
- Aumento de datos suave: volteo horizontal, rotacion pequena y ajuste leve de brillo/contraste.

## Arquitecturas

### BaseCNN

Arquitectura secuencial simple con tres bloques `Conv2d + ReLU + MaxPool2d` y una cabeza completamente conectada. Sirve como baseline interpretable para medir si el pipeline de datos y entrenamiento funciona correctamente.

### AdvancedCNN

Modelo de mayor profundidad con bloques de dos convoluciones, `BatchNorm2d`, `ReLU`, `MaxPool2d` y `Dropout`. La normalizacion por lotes estabiliza el entrenamiento y el dropout reduce sobreajuste, algo especialmente relevante en datasets medicos con posible desbalance y variabilidad limitada.

### TransferModel

Modelo basado en `resnet50` pre-entrenada. Reutiliza representaciones visuales aprendidas en ImageNet y reemplaza la ultima capa por una salida de dos clases. Es el enfoque mas competitivo cuando el dataset disponible no es lo bastante grande para entrenar redes profundas desde cero.

## Estructura del repositorio

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- main.py
|-- models/
|   `-- .gitkeep
`-- src/
    |-- __init__.py
    |-- data/
    |   |-- __init__.py
    |   |-- config.py
    |   |-- constants.py
    |   |-- dataloaders.py
    |   |-- dataset.py
    |   `-- transforms.py
    |-- models/
    |   |-- __init__.py
    |   |-- advanced_cnn.py
    |   |-- base_cnn.py
    |   |-- registry.py
    |   `-- transfer_model.py
    `-- train/
        |-- __init__.py
        |-- cli.py
        |-- device.py
        |-- engine.py
        |-- optimizers.py
        `-- visualization.py
```

Durante el entrenamiento se crea la carpeta:

```text
models/
  best_model.pth
  training_curves.png
```

## Configuracion del entorno

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Si tienes GPU NVIDIA, instala la version de PyTorch compatible con tu version de CUDA siguiendo la guia oficial de PyTorch. El script detecta CUDA automaticamente cuando esta disponible.

## Ejecucion del entrenamiento

Entrenar la CNN base:

```bash
python main.py --data-dir data --model base --epochs 10 --batch-size 32
```

Entrenar la CNN avanzada:

```bash
python main.py --data-dir data --model advanced --epochs 15 --batch-size 32 --learning-rate 0.0001
```

Entrenar con ResNet-50 pre-entrenada:

```bash
python main.py --data-dir data --model transfer --epochs 10 --batch-size 16 --learning-rate 0.0001
```

Congelar el backbone de ResNet-50 y entrenar solo la capa final:

```bash
python main.py --data-dir data --model transfer --freeze-backbone --epochs 5
```

Tambien puedes ejecutar el paquete directamente:

```bash
python -m src.train.cli --data-dir data --model base
```

## Argumentos principales

- `--data-dir`: ruta al dataset con carpetas `train`, `val` y opcionalmente `test`.
- `--model`: arquitectura a usar: `base`, `advanced` o `transfer`.
- `--epochs`: numero de epocas.
- `--batch-size`: tamano del lote.
- `--learning-rate`: tasa de aprendizaje para Adam.
- `--image-size`: tamano de entrada, por defecto `224`.
- `--freeze-backbone`: congela ResNet-50 excepto la capa final.
- `--output-dir`: carpeta de salida, por defecto `models`.

## Agregar, cambiar o eliminar modelos

Cada arquitectura esta aislada en su propio archivo dentro de `src/models/`.

- `src/models/base_cnn.py`: contiene solo `BaseCNN`.
- `src/models/advanced_cnn.py`: contiene solo `AdvancedCNN`.
- `src/models/transfer_model.py`: contiene solo `TransferModel`.

El entrenamiento no importa esas clases directamente. Usa `src/models/registry.py`, que carga el modelo elegido de forma dinamica. Para agregar un modelo nuevo:

1. Crea un archivo nuevo en `src/models/`, por ejemplo `efficient_cnn.py`.
2. Implementa una clase que herede de `torch.nn.Module`.
3. Registra el nombre, modulo y clase en `MODEL_REGISTRY`.

Si eliminas un archivo de modelo, los otros modelos siguen funcionando. Solo fallara si intentas ejecutar `--model` con el nombre del modelo eliminado.

## Salidas

El script guarda automaticamente el mejor modelo segun exactitud de validacion:

```text
models/best_model.pth
```

Tambien genera una grafica con curvas de perdida y exactitud:

```text
models/training_curves.png
```

Si existe el split `test`, al final imprime `accuracy`, `loss` y un reporte de clasificacion con precision, recall y F1-score para ambas clases.

## Buenas practicas incluidas

- Separacion modular entre datos, modelos y entrenamiento.
- Un archivo independiente por arquitectura dentro de `src/models/`.
- Registro dinamico de modelos en `src/models/registry.py`; el entrenamiento no importa clases concretas.
- Soporte automatico para CPU/GPU con `.to(device)`.
- Transformaciones independientes para entrenamiento y evaluacion.
- Guardado del mejor checkpoint por validacion.
- Codigo orientado a clases y funciones pequenas, alineado con principios SOLID.
- Comentarios explicativos en espanol para facilitar mantenimiento academico.
