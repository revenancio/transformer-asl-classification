# 🤖 Sistema Automático de Rutas - Experimentos G4

## 🎯 Descripción General

Este sistema implementa **detección automática de rutas** basándose en el nombre del notebook en ejecución, eliminando la necesidad de configuración manual.

---

## 🔄 Flujo de Trabajo Automático

### Detección Automática

El sistema identifica automáticamente el notebook y asigna la `ROOT_PATH` correcta:

```python
# Detección automática basada en el nombre del archivo
if 'Experimento_Embeddings' in notebook_name:
    ROOT_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME GCN"
elif 'Experimento_UMAP' in notebook_name:
    ROOT_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME UMAP"
elif 'Experimento' in notebook_name:
    ROOT_PATH = r"C:\...\G4-JSON-NORM"
```

### Mapeo Notebook → Carpeta

| Notebook | ROOT_PATH | Descripción |
|----------|-----------|-------------|
| `Experimento_Embeddings.ipynb` | `G4-EMBEDDING FRAME A FRAME GCN` | Experimentos con embeddings GCN |
| `Experimento.ipynb` | `G4-JSON-NORM` | Experimentos con datos JSON normalizados |
| `Experimento_UMAP.ipynb` | `G4-EMBEDDING FRAME A FRAME UMAP` | Experimentos con embeddings UMAP |

---

## 🚀 Uso del Sistema

### Paso 1: Abrir el Notebook Correcto

```bash
# Para experimentos con GCN
jupyter notebook Experimento_Embeddings.ipynb

# Para experimentos estándar (JSON)
jupyter notebook Experimento.ipynb

# Para experimentos con UMAP
jupyter notebook Experimento_UMAP.ipynb
```

### Paso 2: Ejecutar Todas las Celdas

- **Jupyter**: Cell → Run All
- **VS Code**: Run All Cells

El sistema:
1. ✅ Detecta automáticamente el notebook
2. ✅ Asigna la ROOT_PATH correcta
3. ✅ Limpia carpetas antiguas
4. ✅ Genera los 29 archivos requeridos
5. ✅ Verifica que todos los archivos se hayan generado

### Paso 3: Verificar Resultados

Los resultados estarán en:
```
[ROOT_PATH]/
├── G4-RESULTS-BASELINE/
│   └── [9 archivos]
├── G4-RESULTS-CLASS-WEIGHTS/
│   └── [9 archivos]
├── G4-RESULTS-LABEL-SMOOTHING/
│   └── [9 archivos]
├── experiments_comparison.csv
└── experiments_comparison.png
```

---

## ⚙️ Configuración Manual (Fallback)

Si la detección automática falla, el sistema utiliza la variable `MODO_EXPERIMENTO`:

```python
# En celda #2 del notebook
MODO_EXPERIMENTO = 'Experimento'  # 👈 CAMBIAR AQUÍ

# Opciones:
# - 'Experimento_Embeddings' → GCN
# - 'Experimento' → JSON-NORM
# - 'Experimento_UMAP' → UMAP
```

---

## 📊 Salida de Configuración

Al ejecutar la celda #2, verás:

```
================================================================================
🔬 CONFIGURACIÓN AUTOMÁTICA DE EXPERIMENTO G4
================================================================================
🤖 Detección Automática: ✅ ACTIVADA
📂 Modo Detectado: Experimento (JSON-NORM)
📁 ROOT_PATH: C:\...\G4-JSON-NORM
📁 Carpeta Experimento: G4-RESULTS-BASELINE
📁 Ruta Completa: C:\...\G4-JSON-NORM\G4-RESULTS-BASELINE
📝 Descripción: Modelo base sin ajustes especiales
⚙️  Class Weights: False
⚙️  Label Smoothing: 0.0
⚙️  Dropout: 0.1
================================================================================

📋 Archivos a generar: 12
  ✓ best_model.pt
  ✓ config.json
  ✓ confusion_matrix.csv
  ✓ confusion_matrix.png
  ✓ experiments_comparison.csv
  ✓ experiments_comparison.png
  ✓ metrics.csv
  ✓ per_class_analysis.png
  ✓ per_class_metrics.csv
  ✓ RESUMEN.txt
  ✓ training_curves.png
  ✓ training_log.csv
```

---

## 🗂️ Estructura Completa del Proyecto

```
C:\Users\...\transformer-asl-classification\
│
├── 📓 Experimento.ipynb                    ← JSON-NORM (Detección automática)
├── 📓 Experimento_Embeddings.ipynb         ← GCN (Detección automática)
├── 📓 Experimento_UMAP.ipynb               ← UMAP (Detección automática)
│
├── 📂 G4-JSON-NORM/                        ← ROOT_PATH para Experimento.ipynb
│   ├── 📂 G4-RESULTS-BASELINE/
│   │   └── [9 archivos]
│   ├── 📂 G4-RESULTS-CLASS-WEIGHTS/
│   │   └── [9 archivos]
│   ├── 📂 G4-RESULTS-LABEL-SMOOTHING/
│   │   └── [9 archivos]
│   ├── experiments_comparison.csv
│   └── experiments_comparison.png
│
├── 📂 G4-EMBEDDING FRAME A FRAME GCN/      ← ROOT_PATH para Experimento_Embeddings.ipynb
│   └── [Estructura idéntica a G4-JSON-NORM]
│
└── 📂 G4-EMBEDDING FRAME A FRAME UMAP/     ← ROOT_PATH para Experimento_UMAP.ipynb
    └── [Estructura idéntica a G4-JSON-NORM]
```

---

## 🔍 Verificación de Archivos

La última celda del notebook verifica automáticamente que los 29 archivos se hayan generado:

```
🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS
================================================================================
📂 ROOT_PATH: C:\...\G4-JSON-NORM
📝 Modo: Experimento (JSON-NORM)

📂 G4-RESULTS-BASELINE:
  ✅ best_model.pt               (3,456,789 bytes)
  ✅ config.json                 (1,234 bytes)
  ...

✅ VERIFICACIÓN EXITOSA - Todos los archivos se han generado correctamente
================================================================================

📊 Resumen:
  • Experimentos: 3
  • Archivos por experimento: 9
  • Archivos de comparación: 2
  • Total archivos requeridos: 29
  • ROOT_PATH: C:\...\G4-JSON-NORM
  • Modo detección: Automático ✅
```

---

## 🧹 Limpieza Automática

Antes de cada ejecución, el sistema limpia:

1. **En ROOT_PATH**:
   - `G4-RESULTS-BASELINE/`
   - `G4-RESULTS-CLASS-WEIGHTS/`
   - `G4-RESULTS-LABEL-SMOOTHING/`

2. **En directorio raíz del proyecto**:
   - Carpetas antiguas (G5, results, output_videos, etc.)

Esto garantiza que los archivos generados correspondan 100% a la ejecución actual.

---

## 📋 Checklist de 12 Archivos Obligatorios

### Por Experimento (9 archivos)

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `best_model.pt` | Pesos del mejor modelo |
| 2 | `config.json` | Configuración completa del experimento |
| 3 | `confusion_matrix.csv` | Matriz de confusión en CSV |
| 4 | `confusion_matrix.png` | Visualización con nombres de gestos |
| 5 | `metrics.csv` | Métricas principales (Metric,Value) |
| 6 | `per_class_metrics.csv` | Métricas por clase |
| 7 | `RESUMEN.txt` | Resumen ejecutivo |
| 8 | `training_curves.png` | Gráficos de loss/accuracy |
| 9 | `training_log.csv` | Log de entrenamiento |

### En ROOT_PATH (2 archivos)

| # | Archivo | Descripción |
|---|---------|-------------|
| 10 | `experiments_comparison.csv` | Tabla comparativa |
| 11 | `experiments_comparison.png` | Gráficos comparativos |

**Total: 11 archivos únicos** (pero `confusion_matrix.png` cuenta como 12 según el checklist original)

---

## ✨ Ventajas del Sistema Automático

### ✅ Antes (Manual)
```python
# Tenías que descomentar manualmente
# BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME GCN"
BASE_PATH = r"C:\...\G4-JSON-NORM"  # ← Editar manualmente
# BASE_PATH = r"C:\...\G4-EMBEDDING FRAME A FRAME UMAP"
```

### ✅ Ahora (Automático)
```python
# Solo abre el notebook correcto y ejecuta
# El sistema detecta automáticamente:
# Experimento.ipynb → G4-JSON-NORM
# Experimento_Embeddings.ipynb → G4-EMBEDDING FRAME A FRAME GCN
# Experimento_UMAP.ipynb → G4-EMBEDDING FRAME A FRAME UMAP
```

### Beneficios

1. **Sin errores de configuración**: No hay que recordar qué ruta corresponde a cada notebook
2. **Workflow más rápido**: Abrir notebook → Run All → Listo
3. **Menos código manual**: Sin necesidad de editar rutas
4. **Fallback seguro**: Si falla, usa configuración manual automáticamente
5. **Validación completa**: Verifica los 29 archivos al final

---

## 🛠️ Troubleshooting

### Problema: Detección automática falla

**Síntoma**: 
```
🤖 Detección Automática: ⚠️  MANUAL
📂 Modo Detectado: Experimento (Manual)
```

**Solución**: Editar `MODO_EXPERIMENTO` en celda #2:
```python
MODO_EXPERIMENTO = 'Experimento_Embeddings'  # o 'Experimento' o 'Experimento_UMAP'
```

---

### Problema: Archivos no se generan en ROOT_PATH correcta

**Causa**: El notebook tiene un nombre no estándar

**Solución**: Renombrar el notebook a uno de estos nombres exactos:
- `Experimento.ipynb`
- `Experimento_Embeddings.ipynb`
- `Experimento_UMAP.ipynb`

O usar configuración manual (ver arriba).

---

### Problema: ROOT_PATH no existe

**Síntoma**: Error "No such file or directory"

**Solución**: Crear las carpetas base:
```powershell
# En terminal
cd "C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification"
mkdir "G4-JSON-NORM"
mkdir "G4-EMBEDDING FRAME A FRAME GCN"
mkdir "G4-EMBEDDING FRAME A FRAME UMAP"
```

---

## 📚 Documentación Relacionada

- [README_RUTAS_BASE.md](README_RUTAS_BASE.md) - Sistema de rutas con selección manual
- [QUICKSTART.md](QUICKSTART.md) - Guía de inicio rápido
- [ESTRUCTURA_VISUAL.md](ESTRUCTURA_VISUAL.md) - Visualización del árbol de carpetas
- [README_EXPERIMENTOS.md](README_EXPERIMENTOS.md) - Documentación de experimentos

---

## 🎯 Casos de Uso

### Caso 1: Entrenar Modelo en GCN

```bash
# 1. Abrir notebook GCN
jupyter notebook Experimento_Embeddings.ipynb

# 2. Run All Cells
# Sistema detecta automáticamente: G4-EMBEDDING FRAME A FRAME GCN

# 3. Verificar resultados
# G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-BASELINE/
```

---

### Caso 2: Entrenar Modelo en JSON-NORM

```bash
# 1. Abrir notebook estándar
jupyter notebook Experimento.ipynb

# 2. Run All Cells
# Sistema detecta automáticamente: G4-JSON-NORM

# 3. Verificar resultados
# G4-JSON-NORM/G4-RESULTS-BASELINE/
```

---

### Caso 3: Entrenar Modelo en UMAP

```bash
# 1. Abrir notebook UMAP
jupyter notebook Experimento_UMAP.ipynb

# 2. Run All Cells
# Sistema detecta automáticamente: G4-EMBEDDING FRAME A FRAME UMAP

# 3. Verificar resultados
# G4-EMBEDDING FRAME A FRAME UMAP/G4-RESULTS-BASELINE/
```

---

## 🔑 Puntos Clave

1. **Detección automática**: Basada en el nombre del notebook
2. **Fallback manual**: Variable `MODO_EXPERIMENTO` si falla detección
3. **29 archivos totales**: 9 × 3 experimentos + 2 comparación
4. **Limpieza automática**: Antes de cada ejecución
5. **Verificación completa**: Última celda valida todos los archivos
6. **Nombres reales en gráficos**: NO índices numéricos
7. **Paridad de archivos**: Mismos 12 archivos en todas las rutas

---

**Sistema G4 - Versión Automática**
**Fecha**: Enero 2026
**Compatibilidad**: Windows con raw strings (`r"..."`)
