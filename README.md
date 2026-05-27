# Detección de Neumonía en Radiografías de Tórax

Clasificación binaria de imágenes de radiografías de tórax como **NORMAL** o **NEUMONÍA**, utilizando una Red Neuronal Convolucional diseñada a medida y entrenada desde cero en PyTorch.

> 🎓 Proyecto final para el curso de Deep Learning. Solo con fines educativos — **no reemplaza la evaluación médica ni la validación clínica.**

---

## 📋 Tabla de Contenidos

- [Descripción del proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Cómo ejecutar](#cómo-ejecutar)
- [Resultados](#resultados)
- [Declaración de Uso de IA](#declaración-de-uso-de-ia)
- [Referencias](#referencias)

---

## Descripción del proyecto

Implementamos una CNN pequeña (~321K parámetros) llamada `OptimizedCNN`, diseñada para el dataset **Chest X-Ray (Pneumonia)** de Kaggle. Las decisiones clave fueron:

- **Entrenamiento desde cero** (sin transfer learning), para demostrar que una regularización cuidadosa es suficiente para este dataset.
- **Manejo del desbalance de clases en dos frentes**: pérdida ponderada por clase + `WeightedRandomSampler`.
- **Selección de modelo con criterio clínico**: el mejor checkpoint se elige por el **recall en la clase NEUMONÍA**, no por accuracy — los falsos negativos son lo más costoso.
- **Early stopping** + scheduler `ReduceLROnPlateau`.
- **Aumentación de datos agresiva** (`RandomAffine`, `RandomErasing`) para prevenir el sobreajuste en un dataset relativamente pequeño.

## Dataset

- **Fuente**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle.
- **Tamaño**: 5,856 radiografías frontales de tórax de pacientes pediátricos.
- **Clases**: `NORMAL`, `PNEUMONIA`.
- **Particiones** (tal como las provee la fuente):
  - Entrenamiento: 5,216 imágenes
  - Validación: 16 imágenes
  - Prueba: 624 imágenes

Estructura de carpetas esperada:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> Si se ejecuta en Kaggle, el notebook detecta automáticamente `/kaggle/input` (ver Celda 3).

## Estructura del repositorio

```
.
├── arquitectura_optimizada_neumonia.ipynb   # Notebook principal (pipeline completo)
├── Project_Report.pdf                       # Informe final de 2 páginas
├── README.md                                # Este archivo
├── requirements.txt                         # Dependencias de Python
├── AI_USAGE.md                              # Declaración de uso de IA
└── models/                                  # Generado en tiempo de ejecución
    ├── best_optimized_cnn.pth               # Mejor checkpoint
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── class_distribution.png
    ├── sample_images.png
    └── error_examples.png
```

## Instalación

Requiere **Python 3.10+**. Se recomienda usar un entorno virtual.

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd <nombre-del-repositorio>

# 2. Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate            # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Instalar PyTorch con soporte CUDA
# Ver https://pytorch.org/get-started/locally/ para tu versión específica de CUDA.
```

## Cómo ejecutar

### Opción A — Localmente

1. Descargar el dataset de Kaggle y colocarlo en `./data/` siguiendo la estructura indicada anteriormente.
2. Abrir el notebook:

   ```bash
   jupyter notebook arquitectura_optimizada_neumonia.ipynb
   ```

3. Ejecutar las celdas de arriba hacia abajo. **Omitir la Celda 3** (detección automática exclusiva de Kaggle).

### Opción B — En Kaggle

1. Crear un nuevo notebook y subir `arquitectura_optimizada_neumonia.ipynb`.
2. Agregar el dataset **Chest X-Ray Images (Pneumonia)** como entrada.
3. Habilitar el acelerador GPU en *Opciones del notebook*.
4. Ejecutar todas las celdas (la Celda 3 resuelve automáticamente la ruta del dataset).

### Hiperparámetros

Los parámetros más relevantes se encuentran en la Celda 2:

| Parámetro | Valor por defecto | Notas |
|---|---|---|
| `IMAGE_SIZE` | 224 | Resolución de entrada |
| `BATCH_SIZE` | 32 | Reducir si hay errores de memoria |
| `EPOCHS` | 20 | Límite superior; el early stopping determina el fin real |
| `LEARNING_RATE` | 1e-4 | Adam |
| `WEIGHT_DECAY` | 1e-4 | Regularización L2 |
| `DROPOUT_RATE` | 0.4 | Aplicado en la cabeza clasificadora |
| `PATIENCE` | 5 | Paciencia del early stopping |
| `MONITOR_METRIC` | `"recall_pneumonia"` | o `"val_acc"` |
| `USE_WEIGHTED_SAMPLER` | `True` | Balancea los batches |

## Resultados

Rendimiento aproximado en el conjunto de prueba (varía según la semilla aleatoria):

| Métrica | Valor |
|---|---|
| Accuracy | ~0.89 |
| Recall NEUMONÍA | ~0.96 |
| AUC-ROC | ≥0.93 |

Ver el desglose completo, la matriz de confusión y el análisis de errores en `Project_Report.pdf` y en las secciones §8–§10 del notebook.

---

## Declaración de Uso de IA

Este proyecto fue desarrollado con la asistencia de herramientas de inteligencia artificial. En cumplimiento con los requisitos de transparencia académica, declaramos lo siguiente:

### Herramientas de IA Utilizadas

- **Claude (Anthropic)** — Asistente de modelo de lenguaje grande.

### Propósito de Uso

La asistencia de IA se utilizó en un **rol de apoyo** a lo largo de las distintas etapas del proyecto. El equipo conservó la plena propiedad intelectual sobre las decisiones de diseño, los criterios de selección de modelos y la interpretación de resultados. Cada línea de código fue revisada y validada por el equipo antes de ser incluida en la entrega final.

### Áreas del Proyecto Apoyadas por IA

| Área | Cómo se utilizó la IA |
|---|---|
| **Programación** | Sugerencia de expresiones idiomáticas de PyTorch, refactorización de código repetitivo, y apoyo en la estructuración del `Dataset`, `DataLoader`, el ciclo de entrenamiento y las utilidades de evaluación. |
| **Depuración** | Diagnóstico de discrepancias en las dimensiones de tensores, problemas de configuración del dataloader y errores relacionados con la reproducibilidad. |
| **Documentación** | Redacción de docstrings, contenido del README y el informe del proyecto. La redacción final y la precisión técnica fueron revisadas por el equipo. |
| **Exploración de arquitecturas** | Discusión de las ventajas y desventajas entre transfer learning y entrenamiento desde cero, técnicas de regularización (BatchNorm, Dropout, weight decay) y estrategias para el manejo del desbalance de clases. |
| **Análisis e informes** | Estructuración del EDA, el análisis de errores y el informe final. La interpretación de las métricas y las conclusiones son propias del equipo. |

### Lo que NO Se Delegó a la IA

- **Decisiones conceptuales**: la elección del dataset, el planteamiento del problema, la decisión de optimizar el recall para NEUMONÍA y la estrategia de selección del modelo fueron tomadas por el equipo.
- **Validación de resultados**: cada bloque de código, métrica y afirmación del informe fue verificada de forma independiente por el equipo antes de ser entregada.
- **Análisis crítico**: el análisis de errores y las lecciones aprendidas fueron redactados por el equipo a partir de los resultados reales del modelo.

### Declaración de Responsabilidad

El equipo asume la **plena responsabilidad** del trabajo entregado. Comprendemos todo el pipeline, podemos explicar cada decisión de diseño y somos capaces de reproducir todos los resultados reportados. Las herramientas de IA se utilizaron como una ayuda para la productividad — comparable a consultar documentación, libros de texto o un compañero de estudio — pero nunca como sustituto de la comprensión propia.

*Esta declaración se incluye en cumplimiento con el requisito de Declaración de Uso de IA de los entregables del Proyecto de Deep Learning.*

---

## Referencias

1. Kermany, D. S., et al. (2018). *Identifying medical diagnoses and treatable diseases by image-based deep learning.* **Cell**, 172(5), 1122–1131.
2. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization.* **ICML**.
3. Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* **JMLR**, 15(1).
4. He, K., et al. (2015). *Delving Deep into Rectifiers.* **ICCV**.
5. Documentación oficial de PyTorch — https://pytorch.org/docs/stable/

---

**Nota:** Este proyecto es únicamente con fines educativos.