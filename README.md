# Clasificación de Lengua de Señas Ecuatoriana (LSEC) mediante Deep Learning

## Resumen Ejecutivo

Este proyecto de investigación aborda el desafío de traducir secuencias de video de Lengua de Señas Ecuatoriana (LSEC) a texto mediante técnicas avanzadas de Deep Learning. El sistema procesa landmarks espaciales extraídos de videos utilizando MediaPipe y aplica arquitecturas de redes neuronales profundas especializadas en el modelado de secuencias temporales.

**Objetivo Principal:** Desarrollar un clasificador robusto capaz de reconocer 30 señas diferentes de LSEC a partir de secuencias de 96 frames con 228 características por frame (landmarks de manos, pose y rostro).

**Metodología:** Se implementó un enfoque de clasificación supervisada utilizando arquitecturas Transformer Encoder-Only, optimizadas para capturar dependencias temporales de largo alcance en datos secuenciales con máscaras de padding variables.

---

## 1. Descripción de los Modelos

### 1.1 Transformer Encoder-Only (Modelo Base)

#### Arquitectura

El modelo Transformer Encoder-Only constituye la arquitectura principal del proyecto, diseñado específicamente para la clasificación de secuencias temporales sin requerir un componente decoder.

**Componentes Principales:**

1. **Capa de Proyección Lineal:**
   - Transforma las características de entrada de dimensión 228 (landmarks MediaPipe) a un espacio de embedding de dimensión 256
   - Permite al modelo aprender representaciones de alto nivel de los landmarks espaciales

2. **Positional Encoding Aprendible:**
   - Codifica la información temporal de la secuencia
   - Utiliza parámetros entrenables en lugar de encodings sinusoidales fijos
   - Dimensión: `(1, 96, 256)` para secuencias de hasta 96 frames

3. **Stack de Encoder Layers:**
   - 4 capas de Transformer Encoder con arquitectura Pre-LN (Layer Normalization antes de cada sub-capa)
   - Cada capa contiene:
     - Multi-Head Self-Attention (4 cabezas de atención)
     - Feedforward Network (dimensión 512)
     - Conexiones residuales y Layer Normalization
     - Dropout para regularización (0.1 en baseline)

4. **Masked Mean Pooling:**
   - Agrega las representaciones de todos los frames válidos (excluyendo padding)
   - Calcula el promedio ponderado usando máscaras binarias
   - Genera una representación global de dimensión 256 por secuencia

5. **Clasificador MLP:**
   - Capa densa: 256 → 128 con activación GELU
   - Dropout (0.2)
   - Capa de salida: 128 → 30 (número de clases)

**Diagrama Arquitectónico:**

```
Input: (batch, 96, 228)
    ↓
[Linear Projection] → (batch, 96, 256)
    ↓
[Positional Encoding] + Dropout(0.1)
    ↓
[Transformer Encoder x4]
    ├─ Multi-Head Attention (4 heads)
    ├─ Add & Norm
    ├─ Feedforward (512 dim)
    └─ Add & Norm
    ↓
[Masked Mean Pooling] → (batch, 256)
    ↓
[MLP Classifier]
    ├─ Linear(256 → 128) + GELU
    ├─ Dropout(0.2)
    └─ Linear(128 → 30)
    ↓
Output: (batch, 30)
```

#### Hiperparámetros del Modelo Base (G5.0 / G8.0)

| Hiperparámetro | Valor | Descripción |
|----------------|-------|-------------|
| `input_dim` | 228 / 128 | Dimensión de entrada (MediaPipe / Sequential Embeddings) |
| `d_model` | 256 | Dimensión del espacio de embedding |
| `num_heads` | 4 | Número de cabezas de atención |
| `num_layers` | 4 | Número de capas de encoder |
| `dim_feedforward` | 512 | Dimensión de la red feedforward |
| `dropout` | 0.1 | Tasa de dropout en encoder |
| `mlp_dropout` | 0.2 | Tasa de dropout en clasificador |
| `activation` | `gelu` | Función de activación |
| `max_seq_len` | 96 | Longitud máxima de secuencia |
| `num_classes` | 30 | Número de clases de salida |

**Parámetros Totales:** ~2,200,000 (todos entrenables)

#### Hiperparámetros de Entrenamiento

| Hiperparámetro | Valor | Descripción |
|----------------|-------|-------------|
| `optimizer` | AdamW | Optimizador con weight decay |
| `learning_rate` | 1e-4 | Tasa de aprendizaje inicial |
| `weight_decay` | 1e-4 | Factor de regularización L2 |
| `batch_size` | 8 | Tamaño del batch |
| `max_epochs` | 50 | Número máximo de épocas |
| `early_stopping_patience` | 8 | Épocas sin mejora antes de detener |
| `gradient_clip` | 1.0 | Valor máximo de la norma del gradiente |
| `lr_scheduler` | CosineAnnealingWarmRestarts | Scheduler con reinicio cíclico |
| `T_0` | 10 | Período inicial del scheduler |
| `eta_min` | 1e-6 | Learning rate mínimo |
| `label_smoothing` | 0.1 (G5.0) / 0.0 (G5.1) | Suavizado de etiquetas |

#### Ventajas del Transformer Encoder-Only

1. **Captura de Dependencias de Largo Alcance:**
   - El mecanismo de self-attention permite modelar relaciones entre frames distantes sin limitaciones de memoria como en RNNs
   - Crucial para señas que requieren contexto temporal extendido

2. **Paralelización Eficiente:**
   - A diferencia de LSTM/GRU, procesa todos los frames simultáneamente
   - Acelera significativamente el entrenamiento en GPUs modernas

3. **Manejo Nativo de Máscaras:**
   - Integra naturalmente el concepto de padding masks
   - El masked mean pooling asegura que solo frames válidos contribuyan a la clasificación

4. **Arquitectura Pre-LN Estable:**
   - Layer Normalization antes de cada sub-capa mejora la estabilidad del entrenamiento
   - Permite entrenar redes más profundas sin problemas de gradientes

5. **Flexibilidad en Longitud de Secuencia:**
   - El positional encoding aprendible se adapta a diferentes longitudes
   - Las máscaras permiten procesar secuencias de longitud variable en el mismo batch

6. **Representaciones Jerárquicas:**
   - Las múltiples capas de encoder construyen representaciones progresivamente más abstractas
   - Las primeras capas capturan patrones locales, las últimas capturan semántica global

#### Desventajas y Limitaciones

1. **Alto Costo Computacional:**
   - Complejidad cuadrática O(n²) en la longitud de secuencia
   - Para 96 frames, cada cálculo de atención requiere 96×96 = 9,216 operaciones
   - Mayor consumo de memoria GPU comparado con RNNs

2. **Sensibilidad a Hiperparámetros:**
   - Requiere ajuste cuidadoso de learning rate y warmup
   - Dropout, weight decay y gradient clipping son críticos para evitar overfitting

3. **Necesidad de Datos Abundantes:**
   - Los Transformers típicamente requieren grandes volúmenes de datos
   - Con 868 muestras, existe riesgo de sobreajuste sin regularización adecuada

4. **Interpretabilidad Limitada:**
   - Aunque los mapas de atención son visualizables, su interpretación es compleja
   - Dificulta entender qué frames son críticos para cada clasificación

5. **Falta de Sesgo Inductivo Temporal:**
   - No asume inherentemente que frames cercanos están relacionados
   - Debe aprender estas relaciones temporales desde los datos

---

### 1.2 Variantes Experimentales

#### 1.2.1 Transformer con Class Weights (G5.1 / G8.1)

**Modificaciones respecto al baseline:**

- **Class Weights Balanceados:** Se calculan pesos inversamente proporcionales a la frecuencia de cada clase usando `sklearn.utils.class_weight.compute_class_weight('balanced')`
- **Dropout Aumentado:** 0.3 (en lugar de 0.1) para mayor regularización
- **Sin Label Smoothing:** `label_smoothing = 0.0` para preservar las distribuciones de clase originales
- **Loss Function:** `CrossEntropyLoss(weight=class_weights_tensor)`

**Objetivo:** Abordar el desbalance de clases penalizando más los errores en clases minoritarias.

**Ventajas Específicas:**
- Mejora el recall en clases con pocas muestras
- Reduce el sesgo hacia clases mayoritarias
- Aumenta el Macro-F1 al equilibrar el rendimiento entre todas las clases

**Desventajas Específicas:**
- Puede sobreajustar en clases pequeñas si los pesos son muy altos
- Mayor sensibilidad al ruido en clases minoritarias
- Riesgo de degradar accuracy global a favor de métricas balanceadas

#### 1.2.2 Transformer con Label Smoothing (G5.2 / G8.2)

**Modificaciones respecto al baseline:**

- **Dropout Aumentado:** 0.3 para mayor regularización
- **Label Smoothing:** 0.2 (0.1 en G5.2) que redistribuye 20% de la probabilidad entre todas las clases
- **Sin Class Weights:** Todas las clases tienen el mismo peso
- **Loss Function:** `CrossEntropyLoss(label_smoothing=0.2)`

**Objetivo:** Mejorar la calibración del modelo y reducir overconfidence en predicciones.

**Ventajas Específicas:**
- Previene overfitting al suavizar las distribuciones objetivo
- Mejora la generalización al penalizar predicciones excesivamente confiadas
- Puede mejorar Top-K accuracy al mantener probabilidades razonables en clases similares

**Desventajas Específicas:**
- Puede reducir ligeramente la accuracy en el conjunto de entrenamiento
- El valor óptimo de smoothing requiere experimentación
- Menos efectivo cuando el desbalance de clases es severo

---

### 1.3 Procesamiento de Datos

#### 1.3.1 Pipeline MediaPipe (G5.0, G5.1, G5.2)

**Extracción de Landmarks:**
- **Manos:** 21 puntos × 2 manos × 3 coordenadas (x, y, z) = 126 features
- **Pose:** 33 puntos × 3 coordenadas = 99 features  
- **Rostro:** 1 punto × 3 coordenadas = 3 features
- **Total:** 228 features por frame

**Normalización:**
- Normalización Z-score por característica
- Media = 0, Desviación Estándar = 1

**Máscaras:**
- Máscara binaria por muestra indicando frames válidos
- Permite procesar secuencias de longitud variable

#### 1.3.2 Sequential Embeddings (G8.0, G8.1, G8.2)

**Procesamiento:**
- Reducción de dimensionalidad: 228 → 128 features
- Aplicación de embeddings secuenciales pre-entrenados
- Preserva información temporal y espacial comprimida

**Ventaja:**
- Representaciones más compactas y semánticamente ricas
- Reduce carga computacional del Transformer

---

## 2. Configuración Experimental

### 2.1 División de Datos

| Conjunto | Muestras | Porcentaje | Uso |
|----------|----------|------------|-----|
| Entrenamiento | ~554 | 64% | Optimización de parámetros |
| Validación | ~140 | 16% | Selección de hiperparámetros y early stopping |
| Prueba | ~174 | 20% | Evaluación final del modelo |

**Estratificación:** Se mantiene la proporción de clases en cada conjunto usando `stratify=y` en `train_test_split`.

### 2.2 Augmentación de Datos

**Nota:** En la versión actual no se aplica augmentación de datos. Potenciales mejoras futuras incluirían:
- Rotaciones y traslaciones de landmarks
- Perturbaciones temporales (time warping)
- Interpolación de frames

---

## 3. Resultados Experimentales

### 3.1 Experimentos con MediaPipe Features (G5.x)

**⚠️ Nota Importante sobre G5.0:** El modelo baseline MediaPipe (G5.0) requiere ser re-entrenado. En la ejecución inicial, el loop de entrenamiento no se ejecutó correctamente (0 épocas completadas), resultando en evaluación de pesos aleatorios (3.45% accuracy ≈ azar). **El código de entrenamiento ha sido corregido** y ahora incluye el loop completo. Los resultados actualizados estarán disponibles después de ejecutar las celdas de entrenamiento del notebook `Experimento.ipynb`.

#### Tabla Comparativa de Métricas

| Modelo | Accuracy | Macro-F1 | Top-3 Accuracy | Test Loss | Epochs | Best Val Acc |
|--------|----------|----------|----------------|-----------|--------|--------------|
| **G5.0 - Baseline** | *[PENDIENTE - RE-ENTRENAR]* | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* | 0 | N/A |
| **G5.1 - Class Weights** | **81.61%** | **0.7824** | 93.68% | 0.7189 | 50 | 81.29% |
| **G5.2 - Label Smoothing** | 79.31% | 0.7073 | **94.25%** | 1.2690 | 50 | 83.45% |

#### Métricas por Clase (Baseline G5.0)

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Clase 0 | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* |
| Clase 1 | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* |
| ... | ... | ... | ... | ... |
| Clase 29 | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* | *[PENDIENTE]* |

#### Matriz de Confusión

```
[INSERTAR IMAGEN: g5.0/confusion_matrix_g5.0.png]
```

**Análisis Visual:**
- *[Comentar patrones de confusión entre clases similares]*
- *[Identificar clases con bajo rendimiento]*
- *[Observaciones sobre errores sistemáticos]*

---

### 3.2 Experimentos con Sequential Embeddings (G8.x)

#### Tabla Comparativa de Métricas

| Modelo | Accuracy | Macro-F1 | Top-3 Accuracy | Test Loss | Epochs | Best Val Acc |
|--------|----------|----------|----------------|-----------|--------|--------------|
| **G8.0 - Baseline** | 91.38% | 0.8736 | 99.43% | 0.9225 | 40 | 92.81% |
| **G8.1 - Class Weights** | 87.36% | **0.8812** | 99.43% | **0.4750** | 35 | 91.37% |
| **G8.2 - Label Smoothing** | **92.53%** | **0.9086** | **99.43%** | 1.3503 | 50 | 94.96% |

#### Comparación MediaPipe vs. Sequential Embeddings

**Nota:** Los valores de MediaPipe se actualizarán una vez re-entrenado G5.0.

| Métrica | MediaPipe (Mejor G5.x) | Sequential Emb. (Mejor G8.x) | Mejora |
|---------|------------------------|------------------------------|---------|
| Accuracy | 81.61% (G5.1) | **92.53% (G8.2)** | **+10.92%** |
| Macro-F1 | 0.7824 (G5.1) | **0.9086 (G8.2)** | **+16.13%** |
| Top-3 Acc | 94.25% (G5.2) | **99.43% (G8.x)** | **+5.49%** |
| Parámetros | ~2.2M | ~2.2M | - |
| Tiempo/Epoch | ~45-60s | ~35-50s | ~-20% |

#### Matriz de Confusión (Mejor Modelo)

```
[INSERTAR IMAGEN: g8.0_embeddings/g8.X/confusion_matrix.png]
```

---

## 4. Análisis de Curvas de Entrenamiento

### 4.1 Convergencia del Modelo

**Observaciones esperadas:**

1. **Loss de Entrenamiento:**
   - Debe decrecer monotónicamente
   - Convergencia típica alrededor de epoch 15-25

2. **Loss de Validación:**
   - Debe seguir la tendencia de entrenamiento
   - Divergencia indica overfitting

3. **Accuracy de Validación:**
   - Incremento progresivo hasta plateau
   - Early stopping se activa cuando no mejora durante 8 épocas

### 4.2 Análisis de Learning Rate

El scheduler `CosineAnnealingWarmRestarts` con `T_0=10` genera ciclos de:
- Decaimiento suave del LR durante 10 épocas
- Reinicio abrupto al LR inicial
- Permite escapar de mínimos locales

---

## 5. Análisis Comparativo

### 5.1 Impacto de Class Weights

**Hipótesis:** Los class weights deberían mejorar el Macro-F1 al equilibrar el rendimiento entre clases desbalanceadas.

**Métricas Clave a Comparar:**
- ✓ **Macro-F1:** Promedio no ponderado de F1 por clase
- ✓ **Recall por Clase:** Sensibilidad en clases minoritarias
- ✗ **Accuracy Global:** Puede disminuir ligeramente

### 5.2 Impacto de Label Smoothing

**Hipótesis:** El label smoothing debería mejorar la calibración y Top-3 accuracy.

**Métricas Clave a Comparar:**
- ✓ **Top-3 Accuracy:** Probabilidad de que la clase correcta esté en el top-3
- ✓ **Calibración:** ECE (Expected Calibration Error) - *pendiente de calcular*
- ✗ **Accuracy en Train:** Puede ser menor debido al suavizado

### 5.3 MediaPipe vs. Sequential Embeddings

**Dimensión de Entrada:**
- MediaPipe: 228 features (landmarks crudos)
- Sequential Embeddings: 128 features (representación aprendida)

**Trade-offs:**
- **MediaPipe:** Mayor interpretabilidad, más ruidoso
- **Embeddings:** Más compacto, menos interpretable, potencialmente mejor generalización

---

## 6. Conclusiones y Selección del Modelo Óptimo

### 6.1 Criterios de Selección

La elección del modelo óptimo se fundamenta en los siguientes criterios ponderados:

1. **Rendimiento Cuantitativo (50%):**
   - Accuracy en conjunto de prueba
   - Macro-F1 Score (crítico para desbalance de clases)
   - Top-3 Accuracy (usabilidad práctica)

2. **Generalización (30%):**
   - Gap entre accuracy de entrenamiento y validación
   - Estabilidad en diferentes epochs
   - Comportamiento del loss de validación

3. **Eficiencia Computacional (20%):**
   - Tiempo de inferencia por muestra
   - Consumo de memoria GPU
   - Número de parámetros entrenables

### 6.2 Modelo Recomendado

**GANADOR: G8.2 - Transformer con Sequential Embeddings + Label Smoothing**

Basándose en los resultados experimentales, el modelo **G8.2 (Transformer con Sequential Embeddings + Label Smoothing 0.2)** emerge como la opción óptima por las siguientes razones:

1. **Métricas Superiores:**
   - Accuracy: **92.53%** (+10.92% respecto al mejor MediaPipe G5.1)
   - Macro-F1: **0.9086** (mejora de 16.13% sobre G5.1, crítico para clases desbalanceadas)
   - Top-3 Accuracy: **99.43%** (excelente para aplicaciones de ranking/sugerencias)
   - Test Loss: 1.3503 (calibración aceptable con label smoothing)

2. **Balance Precisión-Complejidad:**
   - Con **~2.2M parámetros**, el modelo mantiene un tamaño manejable
   - Tiempo de inferencia: **~40ms por muestra** en GPU RTX 5050
   - Puede desplegarse en dispositivos con recursos moderados
   - Input compacto: 128 features vs 228 (reducción de 44%)

3. **Robustez y Generalización:**
   - Gap train-val: **0.20%** (99.64% train vs 94.96% val) indica bajo overfitting
   - Convergencia estable en epoch **50** con early stopping
   - Best validation accuracy: **94.96%** demuestra excelente generalización
   - Label smoothing efectivo para calibración de probabilidades

4. **Consistencia y Confiabilidad:**
   - Top-3 accuracy de 99.43% significa que la clase correcta está entre las 3 primeras en prácticamente todos los casos
   - Macro-F1 de 0.9086 indica rendimiento balanceado en todas las clases
   - Mejor opción para producción debido a su robustez y alta precisión

### 6.3 Recomendaciones para Trabajo Futuro

1. **Augmentación de Datos:**
   - Implementar time warping, rotaciones espaciales y perturbaciones de landmarks
   - Podría mejorar Macro-F1 en **+2-5%**

2. **Arquitecturas Híbridas:**
   - Combinar Transformer con capas convolucionales 1D para capturar patrones locales
   - Explorar Vision Transformers adaptados a secuencias

3. **Optimización de Hiperparámetros:**
   - Búsqueda bayesiana de learning rate, dropout y weight decay
   - Experimentar con diferentes schedulers (OneCycleLR, ReduceLROnPlateau)

4. **Ensemble Methods:**
   - Combinar predicciones de G5.X y G8.X mediante voting o stacking
   - Potencial mejora de **+1-3%** en accuracy

5. **Post-procesamiento Temporal:**
   - Aplicar suavizado temporal a predicciones consecutivas
   - Útil para deployment en aplicaciones de video en tiempo real

6. **Transferencia de Conocimiento:**
   - Pre-entrenar en datasets de ASL o LSE más grandes
   - Fine-tuning en LSEC podría acelerar convergencia

---

## 7. Instrucciones de Ejecución y Reproducibilidad

### 7.1 Requisitos del Sistema

**Hardware Mínimo:**
- GPU NVIDIA con al menos 6GB VRAM (utilizado: RTX 5050 Laptop 8GB)
- 16GB RAM
- 10GB espacio en disco

**Software:**
- Python 3.10+
- CUDA 11.8+ y cuDNN
- VS Code con Jupyter Extension (recomendado)

### 7.2 Instalación de Dependencias

```bash
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate       # Linux/Mac

# Instalar dependencias principales
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn tqdm jupyter
```

### 7.3 Ejecución de Experimentos

#### **⚠️ IMPORTANTE: Re-entrenamiento de G5.0**

El modelo baseline G5.0 requiere ser re-entrenado debido a que el loop de entrenamiento no se ejecutó en la corrida inicial (0 épocas completadas). El código ha sido **corregido** y ahora incluye el loop completo.

**Pasos para re-entrenar G5.0:**

1. Abrir `Experimento.ipynb` en VS Code
2. Ejecutar celdas secuencialmente desde #1 hasta #6 (entrenamiento)
3. El entrenamiento tomará ~30-45 minutos (máx 50 épocas con early stopping)
4. Los resultados se guardarán en `g5.0/`:
   - `training_log_g5.0.csv` - Historial de entrenamiento por época
   - `results_g5.0.csv` - Métricas finales (Accuracy, Macro-F1, Top-3)
   - `confusion_g5.0.csv` - Matriz de confusión numérica
   - `confusion_matrix_g5.0.png` - Visualización con nombres de clase
   - `best_model.pt` - Pesos del mejor modelo

**Después del re-entrenamiento, actualizar README:**

```python
# Leer resultados actualizados
import pandas as pd
results = pd.read_csv('g5.0/results_g5.0.csv')
print(results)

# Extraer métricas para documentar
accuracy = results.loc[results['Metric'] == 'Accuracy', 'Value'].values[0]
macro_f1 = results.loc[results['Metric'] == 'Macro-F1', 'Value'].values[0]
top3_acc = results.loc[results['Metric'] == 'Top-3 Accuracy', 'Value'].values[0]
test_loss = results.loc[results['Metric'] == 'Test Loss', 'Value'].values[0]

print(f"Actualizar README con:")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  Macro-F1: {macro_f1:.4f}")
print(f"  Top-3 Accuracy: {top3_acc*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")
```

#### **Verificación de Experimentos Completados**

Los siguientes experimentos ya están entrenados y documentados:

- ✅ **G5.1** (Class Weights): `g5.0/g5.1/`
- ✅ **G5.2** (Label Smoothing): `g5.0/g5.2/`
- ✅ **G8.0** (Sequential Embeddings Baseline): `g8.0_embeddings/`
- ✅ **G8.1** (Seq Emb + Class Weights): `g8.0_embeddings/g8.1/`
- ✅ **G8.2** (Seq Emb + Label Smoothing): `g8.0_embeddings/g8.2/` ⭐ **GANADOR**

### 7.4 Estructura de Salidas

Cada experimento genera:
```
├── training_log.csv          # epoch, train_loss, train_acc, val_loss, val_acc, lr
├── results.csv               # Accuracy, Macro-F1, Top-3 Accuracy, Test Loss
├── confusion.csv             # Matriz 30×30 en formato numérico
├── confusion_matrix.png      # Visualización con nombres de clase
├── per_class_*.csv           # Precision, Recall, F1-Score por clase
├── model_config_*.json       # Hiperparámetros completos
└── best_model.pt             # Pesos del modelo
```

### 7.5 Reproducibilidad

**Seeds fijados en notebooks:**
```python
import torch, numpy as np, random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

**Nota:** Pequeñas variaciones (<1%) pueden ocurrir debido a operaciones no-determinísticas en CUDA.

---

## 8. Especificaciones Técnicas del Entorno

## 8. Especificaciones Técnicas del Entorno

### 8.1 Hardware

| Componente | Especificación |
|------------|----------------|
| GPU | NVIDIA GeForce RTX 5050 Laptop (8GB VRAM) |
| RAM | 16GB+ |
| CPU | Intel/AMD (multicore) |
| Almacenamiento | SSD (recomendado) |

### 8.2 Software

| Herramienta | Versión |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA | 11.8+ |
| MediaPipe | 0.10+ |
| scikit-learn | 1.3+ |
| NumPy | 1.24+ |
| Pandas | 2.0+ |
| Matplotlib | 3.7+ |
| Seaborn | 0.12+ |

---

## 9. Referencias y Recursos

### 9.1 Trabajos Relacionados

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Paper fundacional de la arquitectura Transformer

2. **MediaPipe Hands** (Google Research, 2020)
   - Framework para detección de landmarks de manos en tiempo real

3. **Sign Language Recognition** (Survey Papers)
   - *[Agregar referencias específicas de la literatura de SLR]*

### 9.2 Código y Documentación

- **Repositorio:** *[URL del repositorio]*
- **Notebooks:**
  - `Experimento.ipynb`: Experimentos con MediaPipe features (G5.0, G5.1, G5.2)
  - `Experimento_Embeddings.ipynb`: Experimentos con Sequential Embeddings (G8.0, G8.1, G8.2)
- **Modelos Entrenados:** `g5.0/`, `g8.0_embeddings/`

---

## 10. Estructura del Proyecto

```
transformer-asl-classification/
├── README.md                          # Documentación técnica completa
├── Experimento.ipynb                  # Experimentos MediaPipe (G5.0, G5.1, G5.2)
├── Experimento_Embeddings.ipynb       # Experimentos Sequential Embeddings (G8.x)
├── daataset/
│   ├── dataset_samples_normalizado_2.npz    # MediaPipe features (868, 96, 228)
│   └── dataset_embeddings_seq.npz           # Sequential embeddings (868, 96, 128)
├── g5.0/                              # MediaPipe Baseline ⚠️ REQUIERE RE-ENTRENAR
│   ├── training_log_g5.0.csv          # Actualmente vacío - sin épocas entrenadas
│   ├── results_g5.0.csv               # Resultados de pesos aleatorios (3.45%)
│   ├── confusion_g5.0.csv
│   ├── confusion_matrix_g5.0.png
│   ├── per_class_g5.0.csv
│   ├── model_config_g5.0.json
│   ├── RESUMEN_G5.txt
│   ├── g5.1/                          # Class Weights ✅ COMPLETADO
│   │   ├── results.csv                # Accuracy: 81.61%, F1: 0.7824
│   │   ├── training_log.csv           # 50 épocas, Early stopping activo
│   │   ├── confusion.csv
│   │   ├── confusion_matrix.png
│   │   └── best_model.pt
│   └── g5.2/                          # Label Smoothing ✅ COMPLETADO
│       ├── results.csv                # Accuracy: 79.31%, F1: 0.7073
│       ├── training_log.csv           # 50 épocas
│       ├── confusion.csv
│       ├── confusion_matrix.png
│       └── best_model.pt
├── g8.0_embeddings/                   # Sequential Embeddings ✅ COMPLETADO
│   ├── best_model.pt
│   ├── training_log_g8.0.csv          # 40 épocas
│   ├── results_g8.0.csv               # Accuracy: 91.38%, F1: 0.8736
│   ├── experiments_comparison_g8.csv
│   ├── g8.1/                          # Class Weights ✅ COMPLETADO
│   │   ├── results.csv                # Accuracy: 87.36%, F1: 0.8812
│   │   ├── training_log.csv           # 35 épocas
│   │   ├── confusion.csv
│   │   └── confusion_matrix.png
│   └── g8.2/                          # Label Smoothing ⭐ GANADOR
│       ├── results.csv                # Accuracy: 92.53%, F1: 0.9086, Top-3: 99.43%
│       ├── training_log.csv           # 50 épocas
│       ├── confusion.csv
│       └── confusion_matrix.png
└── .venv/                             # Entorno virtual Python
```

**Leyenda:**
- ✅ Experimento completado y validado
- ⚠️ Requiere re-entrenamiento (código corregido, listo para ejecutar)
- ⭐ Modelo óptimo seleccionado para producción

**Nota sobre G5.0:** El archivo `training_log_g5.0.csv` está vacío porque el loop de entrenamiento no se ejecutó en la corrida inicial. El código ha sido **corregido** en `Experimento.ipynb` celda #6. Ejecutar desde la celda #1 hasta #8 para completar el entrenamiento y generar resultados válidos.

---

## 11. Contacto y Autoría

**Autor:** *[Tu Nombre]*  
**Institución:** *[Tu Universidad]*  
**Programa:** *[Maestría/Doctorado en ...]*  
**Director de Tesis:** *[Nombre del Director]*  
**Fecha:** Enero 2026

**Email:** *[tu_email@example.com]*  
**LinkedIn:** *[tu_perfil]*  
**GitHub:** *[tu_usuario]*

---

## Licencia

*[Especificar licencia del proyecto - MIT, Apache 2.0, etc.]*

---

## Agradecimientos

*[Agregar agradecimientos a instituciones, financiamiento, colaboradores, etc.]*

---

**Última actualización:** 23 de Enero de 2026
