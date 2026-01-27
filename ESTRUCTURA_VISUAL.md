# 📁 Sistema de Rutas Base Absolutas - Estructura Visual

## 🌳 Estructura del Proyecto Completa

```
C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\
│
├── 📓 Experimento.ipynb                          ← Notebook principal (MODIFICADO)
├── 📄 README.md
├── 📄 README_EXPERIMENTOS.md
├── 📄 README_RUTAS_BASE.md                       ← Guía de rutas (NUEVO)
├── 📄 QUICKSTART.md                              ← Actualizado con rutas absolutas
│
├── 📂 daataset/
│   ├── dataset_embeddings_seq.npz
│   ├── dataset_samples_normalizado_2.npz
│   └── 📂 frame to frame/
│       ├── class_names.npy                       ← Nombres de gestos ASL (IMPORTANTE)
│       ├── masks.npy
│       ├── X.npy
│       └── y.npy
│
├── 📂 G4-EMBEDDING FRAME A FRAME GCN/            ← OPCIÓN 1 (BASE_PATH)
│   ├── best_model.pt                             ← Archivos del proyecto GCN original
│   ├── confusion_g8.0.csv
│   ├── experiments_comparison_g8.csv
│   ├── model_config_g8.0.json
│   ├── per_class_g8.0.csv
│   ├── results_g8.0.csv
│   ├── RESUMEN_G8.0.txt
│   ├── training_log_g8.0.csv
│   │
│   └── 📂 [NUEVOS RESULTADOS - Generados por notebook] 
│       ├── 📂 G4-RESULTS-BASELINE/               ← Experimento 1 (9 archivos)
│       │   ├── best_model.pt
│       │   ├── config.json
│       │   ├── confusion_matrix.csv
│       │   ├── confusion_matrix.png
│       │   ├── metrics.csv
│       │   ├── per_class_metrics.csv
│       │   ├── RESUMEN.txt
│       │   ├── training_curves.png
│       │   └── training_log.csv
│       │
│       ├── 📂 G4-RESULTS-CLASS-WEIGHTS/          ← Experimento 2 (9 archivos)
│       │   ├── best_model.pt
│       │   ├── config.json
│       │   ├── confusion_matrix.csv
│       │   ├── confusion_matrix.png
│       │   ├── metrics.csv
│       │   ├── per_class_metrics.csv
│       │   ├── RESUMEN.txt
│       │   ├── training_curves.png
│       │   └── training_log.csv
│       │
│       ├── 📂 G4-RESULTS-LABEL-SMOOTHING/        ← Experimento 3 (9 archivos)
│       │   ├── best_model.pt
│       │   ├── config.json
│       │   ├── confusion_matrix.csv
│       │   ├── confusion_matrix.png
│       │   ├── metrics.csv
│       │   ├── per_class_metrics.csv
│       │   ├── RESUMEN.txt
│       │   ├── training_curves.png
│       │   └── training_log.csv
│       │
│       ├── experiments_comparison.csv            ← Comparación de 3 experimentos
│       └── experiments_comparison.png
│
├── 📂 G4-EMBEDDING FRAME A FRAME UMAP/           ← OPCIÓN 2 (BASE_PATH)
│   ├── confusion_umap.csv                        ← Archivos del proyecto UMAP original
│   ├── experiments_comparison_umap.csv
│   ├── model_config_umap.json
│   ├── per_class_umap.csv
│   ├── results_umap.csv
│   ├── RESUMEN_UMAP.txt
│   ├── training_log_umap.csv
│   │
│   └── 📂 [NUEVOS RESULTADOS - Estructura idéntica a GCN]
│       ├── 📂 G4-RESULTS-BASELINE/
│       ├── 📂 G4-RESULTS-CLASS-WEIGHTS/
│       ├── 📂 G4-RESULTS-LABEL-SMOOTHING/
│       ├── experiments_comparison.csv
│       └── experiments_comparison.png
│
└── 📂 G4-JSON-NORM/                              ← OPCIÓN 3 (BASE_PATH)
    ├── best_model.pt                             ← Archivos del proyecto JSON original
    ├── confusion_g5.0.csv
    ├── experiments_comparison.csv
    ├── model_config_g5.0.json
    ├── model_weights.pt
    ├── per_class_g5.0.csv
    ├── results_g5.0.csv
    ├── RESUMEN_G5.txt
    ├── training_log_g5.0.csv
    │
    └── 📂 [NUEVOS RESULTADOS - Estructura idéntica a GCN y UMAP]
        ├── 📂 G4-RESULTS-BASELINE/
        ├── 📂 G4-RESULTS-CLASS-WEIGHTS/
        ├── 📂 G4-RESULTS-LABEL-SMOOTHING/
        ├── experiments_comparison.csv
        └── experiments_comparison.png
```

---

## 🎯 Flujo de Ejecución

### Configuración en Celda #2

```python
# ═══════════════════════════════════════════════════════════════════════════
# PASO 1: SELECCIONAR RUTA BASE (Descomentar la ruta deseada)
# ═══════════════════════════════════════════════════════════════════════════

# Opción 1: GCN con Embeddings
BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-EMBEDDING FRAME A FRAME GCN"
                    ↓
        [Los resultados se guardan AQUÍ]
                    ↓
    G4-EMBEDDING FRAME A FRAME GCN/
    ├── G4-RESULTS-BASELINE/
    ├── G4-RESULTS-CLASS-WEIGHTS/
    └── G4-RESULTS-LABEL-SMOOTHING/
```

---

## 📊 Conteo de Archivos

### Por Experimento Individual

```
G4-RESULTS-BASELINE/
├── 1.  best_model.pt              (Modelo entrenado)
├── 2.  config.json                (Configuración)
├── 3.  confusion_matrix.csv       (Matriz CSV)
├── 4.  confusion_matrix.png       (Matriz visualizada)
├── 5.  metrics.csv                (Métricas principales)
├── 6.  per_class_metrics.csv      (Métricas por clase)
├── 7.  RESUMEN.txt                (Resumen ejecutivo)
├── 8.  training_curves.png        (Curvas de aprendizaje)
└── 9.  training_log.csv           (Log de entrenamiento)

Total: 9 archivos
```

### Completo (3 Experimentos)

```
[BASE_PATH]/
├── G4-RESULTS-BASELINE/           (9 archivos)
├── G4-RESULTS-CLASS-WEIGHTS/      (9 archivos)
├── G4-RESULTS-LABEL-SMOOTHING/    (9 archivos)
├── experiments_comparison.csv     (1 archivo)
└── experiments_comparison.png     (1 archivo)

Total: 9 × 3 + 2 = 29 archivos
```

---

## 🔄 Tres Modos de Uso

### Modo 1: Experimento Individual en GCN

```python
# Celda #2
BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME GCN"
EXPERIMENT_TYPE = 'baseline'

# Ejecutar celdas 1-17
# Resultado: 9 archivos en G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-BASELINE/
```

---

### Modo 2: Experimento Individual en UMAP

```python
# Celda #2
BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME UMAP"
EXPERIMENT_TYPE = 'class_weights'

# Ejecutar celdas 1-17
# Resultado: 9 archivos en G4-EMBEDDING FRAME A FRAME UMAP/G4-RESULTS-CLASS-WEIGHTS/
```

---

### Modo 3: Comparación Completa en JSON-NORM

```python
# Celda #2
BASE_PATH = r"C:\...\G4-JSON-NORM"
# EXPERIMENT_TYPE se ignora (se ejecutan los 3)

# Ejecutar TODAS las celdas
# Resultado: 29 archivos en G4-JSON-NORM/
```

---

## 🎨 Características de Visualizaciones

### Matriz de Confusión (confusion_matrix.png)

```
✅ ANTES (índices numéricos):
    0    1    2    3   ...
0  [50]  [2]  [1]  [0]
1  [3] [45]  [2]  [1]
...

✅ AHORA (nombres reales):
              hola  gracias  por favor  adiós  ...
hola          [50]    [2]      [1]      [0]
gracias       [3]    [45]      [2]      [1]
...
```

### Análisis por Clase (per_class_analysis.png)

```
✅ ANTES:
Clase 0: F1 = 0.85
Clase 1: F1 = 0.90
...

✅ AHORA:
hola:       F1 = 0.85 ████████████████▌
gracias:    F1 = 0.90 █████████████████▌
por favor:  F1 = 0.78 ██████████████▌
...
```

---

## 🛡️ Validación de Archivos

### Celda de Verificación (última celda)

```
🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS
================================================================================

📂 G4-RESULTS-BASELINE:
  ✅ best_model.pt               (3,456,789 bytes)
  ✅ config.json                 (1,234 bytes)
  ✅ confusion_matrix.csv        (5,678 bytes)
  ✅ confusion_matrix.png        (234,567 bytes)
  ✅ metrics.csv                 (156 bytes)
  ✅ per_class_metrics.csv       (3,456 bytes)
  ✅ RESUMEN.txt                 (2,345 bytes)
  ✅ training_curves.png         (187,654 bytes)
  ✅ training_log.csv            (987 bytes)

📂 G4-RESULTS-CLASS-WEIGHTS:
  ✅ best_model.pt               (3,456,789 bytes)
  ... (9 archivos)

📂 G4-RESULTS-LABEL-SMOOTHING:
  ✅ best_model.pt               (3,456,789 bytes)
  ... (9 archivos)

📂 Archivos de comparación en BASE_PATH:
  ✅ experiments_comparison.csv  (456 bytes)
  ✅ experiments_comparison.png  (123,456 bytes)

================================================================================
✅ VERIFICACIÓN EXITOSA - Todos los archivos se han generado correctamente
================================================================================

📊 Resumen:
  • Experimentos: 3
  • Archivos por experimento: 9
  • Archivos de comparación: 2
  • Total archivos requeridos: 29
  • Ruta base: C:\...\G4-EMBEDDING FRAME A FRAME GCN
```

---

## 📝 Formato de Archivos Clave

### metrics.csv (Formato ESTRICTO)

```csv
Metric,Value
Accuracy,0.7890
Macro-F1,0.7345
Top-3 Accuracy,0.9123
Test Loss,0.6543
```

### config.json (Extracto)

```json
{
  "experiment_type": "baseline",
  "architecture": "TransformerEncoderOnly",
  "input_dim": 228,
  "d_model": 256,
  "num_heads": 4,
  "num_layers": 4,
  "dropout": 0.1,
  "use_class_weights": false,
  "label_smoothing": 0.0,
  "test_accuracy": 0.7890,
  "test_macro_f1": 0.7345,
  "best_epoch": 23,
  "training_timestamp": "2024-01-15T14:32:10"
}
```

### per_class_metrics.csv (Extracto)

```csv
,precision,recall,f1-score,support
hola,0.85,0.88,0.87,25
gracias,0.90,0.92,0.91,30
por favor,0.78,0.75,0.76,20
adiós,0.82,0.85,0.83,22
...
```

---

## ⚙️ Limpieza Automática

### Celda #3: Limpieza Antes de Ejecutar

```python
# 🧹 LIMPIEZA AUTOMÁTICA DE CARPETAS DE RESULTADOS

# Se eliminan:
# 1. Carpetas de resultados en BASE_PATH:
#    - G4-RESULTS-BASELINE
#    - G4-RESULTS-CLASS-WEIGHTS
#    - G4-RESULTS-LABEL-SMOOTHING

# 2. Carpetas antiguas en directorio raíz:
#    - output_videos
#    - temp_results
#    - old_results
#    - G5-RESULTS-* (versión anterior)
#    - results/
```

---

## 🔑 Puntos Clave

1. **Raw strings obligatorios**: `r"C:\..."` (evita problemas con `\`)
2. **Solo una ruta activa**: Descomentar UNA de las tres opciones
3. **29 archivos totales**: 9 por experimento × 3 + 2 comparación
4. **Nombres reales en gráficos**: NO índices (0, 1, 2...)
5. **Verificación automática**: Última celda valida todos los archivos
6. **Limpieza automática**: Antes de ejecutar, se eliminan carpetas antiguas
7. **Independencia de proyectos**: Cada BASE_PATH tiene sus propios resultados

---

## 📚 Documentación Relacionada

- [README_RUTAS_BASE.md](README_RUTAS_BASE.md) - Guía completa del sistema
- [QUICKSTART.md](QUICKSTART.md) - Inicio rápido actualizado
- [README_EXPERIMENTOS.md](README_EXPERIMENTOS.md) - Documentación de experimentos

---

**Sistema G4 - Versión de Rutas Absolutas**
**Última actualización**: 2024
