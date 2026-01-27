# 📋 Guía de Uso - Sistema de Rutas Base Absolutas (G4)

## 🎯 Descripción General

Este notebook implementa un sistema de **rutas base absolutas** para organizar experimentos de Machine Learning en tres proyectos diferentes:

1. **G4-EMBEDDING FRAME A FRAME GCN** - Experimentos con embeddings GCN
2. **G4-EMBEDDING FRAME A FRAME UMAP** - Experimentos con embeddings UMAP
3. **G4-JSON-NORM** - Experimentos con datos JSON normalizados

## 🗂️ Estructura de Archivos Generados

Cada experimento genera **12 archivos obligatorios** organizados en la siguiente estructura:

```
[RUTA_BASE_SELECCIONADA]/
├── G4-RESULTS-BASELINE/
│   ├── best_model.pt
│   ├── config.json
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── metrics.csv
│   ├── per_class_metrics.csv
│   ├── RESUMEN.txt
│   ├── training_curves.png
│   └── training_log.csv
│
├── G4-RESULTS-CLASS-WEIGHTS/
│   ├── [mismos 9 archivos]
│   └── ...
│
├── G4-RESULTS-LABEL-SMOOTHING/
│   ├── [mismos 9 archivos]
│   └── ...
│
├── experiments_comparison.csv
└── experiments_comparison.png
```

**Total: 29 archivos** (9 por experimento × 3 experimentos + 2 archivos de comparación)

---

## 🚀 Instrucciones de Uso

### Paso 1: Seleccionar Ruta Base

En la celda de configuración (celda #2), **descomentar UNA de las tres rutas**:

```python
# ═══════════════════════════════════════════════════════════════════════════
# PASO 1: SELECCIONAR RUTA BASE (Descomentar la ruta deseada)
# ═══════════════════════════════════════════════════════════════════════════

# Opción 1: GCN con Embeddings
BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-EMBEDDING FRAME A FRAME GCN"

# Opción 2: UMAP con Embeddings
# BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-EMBEDDING FRAME A FRAME UMAP"

# Opción 3: JSON Normalizado
# BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-JSON-NORM"
```

> ⚠️ **IMPORTANTE**: Solo UNA ruta debe estar activa (sin `#` al inicio). Las demás deben estar comentadas.

---

### Paso 2: Seleccionar Tipo de Experimento

En la misma celda, seleccionar el tipo de experimento:

```python
# ═══════════════════════════════════════════════════════════════════════════
# PASO 2: SELECCIONAR TIPO DE EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_TYPE = 'baseline'  # Opciones: 'baseline', 'class_weights', 'label_smoothing'
```

**Opciones disponibles:**

| EXPERIMENT_TYPE     | Carpeta de Salida              | Características                                    |
|---------------------|--------------------------------|----------------------------------------------------|
| `'baseline'`        | `G4-RESULTS-BASELINE`          | Dropout 0.1, sin class weights, sin label smoothing |
| `'class_weights'`   | `G4-RESULTS-CLASS-WEIGHTS`     | Dropout 0.3, con class weights, sin label smoothing |
| `'label_smoothing'` | `G4-RESULTS-LABEL-SMOOTHING`   | Dropout 0.3, sin class weights, con label smoothing 0.1 |

---

### Paso 3: Ejecutar el Notebook

#### Opción A: Ejecutar Experimento Individual

1. Configurar `EXPERIMENT_TYPE` (ej: `'baseline'`)
2. Ejecutar todas las celdas hasta la celda #17 (antes de "Experimentos de Mejora")
3. Resultado: Se generarán 9 archivos en `[BASE_PATH]/G4-RESULTS-BASELINE/`

#### Opción B: Ejecutar los 3 Experimentos Completos

1. Ejecutar **TODAS las celdas del notebook** (incluye celdas #21, #23, #25)
2. Resultado: Se generarán 29 archivos totales:
   - 9 archivos en `G4-RESULTS-BASELINE/`
   - 9 archivos en `G4-RESULTS-CLASS-WEIGHTS/`
   - 9 archivos en `G4-RESULTS-LABEL-SMOOTHING/`
   - 2 archivos de comparación en `[BASE_PATH]/`

---

### Paso 4: Verificar Generación de Archivos

Ejecutar la última celda del notebook (celda de verificación) para validar que todos los archivos se hayan generado:

```python
# 🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS (12 ARCHIVOS OBLIGATORIOS)
```

**Salida esperada:**

```
🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS
================================================================================

📂 G4-RESULTS-BASELINE:
  ✅ best_model.pt               (1,234,567 bytes)
  ✅ config.json                 (1,234 bytes)
  ✅ confusion_matrix.csv        (5,678 bytes)
  ✅ confusion_matrix.png        (123,456 bytes)
  ✅ metrics.csv                 (234 bytes)
  ✅ per_class_metrics.csv       (3,456 bytes)
  ✅ RESUMEN.txt                 (2,345 bytes)
  ✅ training_curves.png         (98,765 bytes)
  ✅ training_log.csv            (1,234 bytes)

...

✅ VERIFICACIÓN EXITOSA - Todos los archivos se han generado correctamente
```

---

## 📊 Descripción de Archivos Generados

### Archivos por Experimento (9 archivos)

| Archivo                     | Descripción                                                    |
|-----------------------------|----------------------------------------------------------------|
| `best_model.pt`             | Pesos del mejor modelo (según validación)                      |
| `config.json`               | Configuración completa del experimento (hiperparámetros)       |
| `confusion_matrix.csv`      | Matriz de confusión en formato CSV                             |
| `confusion_matrix.png`      | Visualización de matriz de confusión con nombres de clases    |
| `metrics.csv`               | Métricas principales (formato: Metric,Value)                   |
| `per_class_metrics.csv`     | Precision, Recall, F1-Score por clase (con nombres de clases) |
| `RESUMEN.txt`               | Resumen ejecutivo del experimento                              |
| `training_curves.png`       | Gráficos de loss, accuracy, learning rate                      |
| `training_log.csv`          | Log completo de entrenamiento (epoch by epoch)                 |

### Archivos de Comparación (2 archivos en BASE_PATH)

| Archivo                       | Descripción                                              |
|-------------------------------|----------------------------------------------------------|
| `experiments_comparison.csv`  | Tabla comparativa de los 3 experimentos                  |
| `experiments_comparison.png`  | Gráficos comparativos de Accuracy, Macro-F1, Top-3 Acc   |

---

## 🛠️ Características Especiales

### ✅ Características Implementadas

1. **Rutas Absolutas con raw strings (`r"..."`)**
   - Evita problemas con caracteres especiales en Windows
   - Maneja espacios en nombres de carpetas

2. **Limpieza Automática Antes de Ejecutar**
   - Elimina carpetas de resultados anteriores
   - Limpia tanto en `BASE_PATH` como en directorio raíz del proyecto

3. **Nombres de Clases en Visualizaciones**
   - **NO usa índices numéricos (0, 1, 2...)**
   - Usa nombres reales de gestos ASL (`class_names.npy`)

4. **Formato Estricto de `metrics.csv`**
   ```csv
   Metric,Value
   Accuracy,0.7890
   Macro-F1,0.7345
   Top-3 Accuracy,0.9123
   Test Loss,0.6543
   ```

5. **Validación de Archivos Generados**
   - Verifica que los 29 archivos existan
   - Muestra tamaño de cada archivo
   - Alerta si falta algún archivo

---

## 🔄 Flujo de Trabajo Típico

### Escenario 1: Entrenar Modelo Baseline en Proyecto GCN

```python
# 1. Configurar ruta base
BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME GCN"

# 2. Configurar experimento
EXPERIMENT_TYPE = 'baseline'

# 3. Ejecutar celdas 1-17
# Resultado: 9 archivos en G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-BASELINE/
```

---

### Escenario 2: Comparar 3 Estrategias en Proyecto UMAP

```python
# 1. Configurar ruta base
BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME UMAP"

# 2. EXPERIMENT_TYPE se ignora (se ejecutan los 3 experimentos)

# 3. Ejecutar TODAS las celdas
# Resultado: 29 archivos totales en G4-EMBEDDING FRAME A FRAME UMAP/
```

---

### Escenario 3: Re-ejecutar Experimento Class Weights

```python
# 1. Configurar ruta base
BASE_PATH = r"C:\...\G4-JSON-NORM"

# 2. Configurar experimento
EXPERIMENT_TYPE = 'class_weights'

# 3. Ejecutar celdas 1-17
# Resultado: 9 archivos en G4-JSON-NORM/G4-RESULTS-CLASS-WEIGHTS/
```

---

## ⚠️ Troubleshooting

### Problema: "No such file or directory"

**Causa**: Ruta base no existe o tiene typo

**Solución**:
1. Verificar que la carpeta exista: `G4-EMBEDDING FRAME A FRAME GCN`, `G4-EMBEDDING FRAME A FRAME UMAP`, o `G4-JSON-NORM`
2. Copiar ruta desde explorador de archivos
3. Usar **raw string**: `r"C:\..."`

---

### Problema: Archivos no se generan en BASE_PATH

**Causa**: Celdas de experimentos 1 y 2 no actualizadas

**Solución**:
- Verificar que las celdas #21, #23, #25 usen `Path(BASE_PATH) / 'G4-RESULTS-...'`
- Re-ejecutar el notebook completo

---

### Problema: Matriz de confusión muestra índices (0, 1, 2...)

**Causa**: Archivo `class_names.npy` no cargado o corrupto

**Solución**:
```python
# Verificar carga de class_names
class_names_path = Path('daataset/frame to frame/class_names.npy')
class_names = np.load(class_names_path, allow_pickle=True)
print(f"Clases cargadas: {len(class_names)}")
print(f"Primeras 5 clases: {class_names[:5]}")
```

---

## 📞 Soporte

Para problemas o preguntas, revisar:

1. [QUICKSTART.md](QUICKSTART.md) - Guía rápida de inicio
2. [README_EXPERIMENTOS.md](README_EXPERIMENTOS.md) - Documentación de experimentos
3. Celda de verificación de archivos (última celda del notebook)

---

## 📝 Notas Importantes

1. **NO modificar nombres de carpetas generadas** (deben seguir formato `G4-RESULTS-*`)
2. **Usar SIEMPRE raw strings** para rutas en Windows: `r"C:\..."`
3. **Ejecutar celda de limpieza** antes de re-ejecutar experimentos
4. **Verificar archivos generados** después de cada ejecución
5. Los archivos de comparación (`experiments_comparison.*`) solo se generan al ejecutar los 3 experimentos completos

---

**Última actualización**: 2024
**Versión del sistema**: G4
**Archivos requeridos**: 29 (9 × 3 experimentos + 2 comparación)
