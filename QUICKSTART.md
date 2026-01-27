# 🚀 INICIO RÁPIDO - Experimentos Transformer ASL (FORMATO G4)

## 📌 Pasos para ejecutar un experimento

### 1️⃣ Abrir el notebook
```bash
jupyter notebook Experimento.ipynb
# O en VS Code: Abrir Experimento.ipynb
```

### 2️⃣ Seleccionar la RUTA BASE (Celda #2)

**PRIMERO**, selecciona dónde quieres guardar los resultados descomentando UNA ruta:

```python
# Opción 1: GCN con Embeddings
BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-EMBEDDING FRAME A FRAME GCN"

# Opción 2: UMAP con Embeddings
# BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-EMBEDDING FRAME A FRAME UMAP"

# Opción 3: JSON Normalizado
# BASE_PATH = r"C:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification\G4-JSON-NORM"
```

> ⚠️ Solo UNA ruta debe estar activa (sin `#`)

### 3️⃣ Seleccionar el tipo de experimento

En la MISMA celda, selecciona el tipo de experimento:

```python
EXPERIMENT_TYPE = 'baseline'  # 👈 CAMBIAR AQUÍ
```

**Opciones disponibles:**
- `'baseline'` - Modelo base (sin ajustes especiales)
- `'class_weights'` - Con balanceo de clases
- `'label_smoothing'` - Con suavizado de etiquetas

### 4️⃣ Ejecutar todo el notebook
- **Jupyter**: Cell → Run All
- **VS Code**: Run All Cells

### 5️⃣ Encontrar los resultados

Los resultados se guardan automáticamente en **FORMATO G4** DENTRO de la ruta base seleccionada:
```
[BASE_PATH]/G4-RESULTS-[TIPO]/
```

Por ejemplo, si seleccionaste la Opción 1 (GCN):
- Baseline: `G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-BASELINE/`
- Class Weights: `G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-CLASS-WEIGHTS/`
- Label Smoothing: `G4-EMBEDDING FRAME A FRAME GCN/G4-RESULTS-LABEL-SMOOTHING/`

---

## 📊 Archivos generados (en cada carpeta de experimento)

| Archivo | Descripción |
|---------|-------------|
| `config.json` | Hiperparámetros completos del experimento |
| `metrics.csv` | **Métricas principales** (Accuracy, F1, Top-3, Loss) |
| `training_log.csv` | Historial de cada época |
| `confusion_matrix.csv` | Matriz de confusión en CSV |
| `confusion_matrix.png` | **Matriz de confusión** con nombres de gestos |
| `training_curves.png` | **Curvas de Loss y Accuracy** |
| `per_class_analysis.png` | **Análisis detallado** por cada gesto |
| `per_class_metrics.csv` | Métricas numéricas por clase |
| `RESUMEN.txt` | Resumen ejecutivo del experimento |
| `best_model.pt` | Pesos del mejor modelo |

**Total por experimento: 9 archivos**

### Archivos de comparación (en BASE_PATH)

| Archivo | Descripción |
|---------|-------------|
| `experiments_comparison.csv` | Tabla comparativa de los 3 experimentos |
| `experiments_comparison.png` | Gráficos comparativos (Accuracy, F1, Top-3) |

**Total archivos de comparación: 2 archivos**

> 📝 Los archivos de comparación solo se generan si ejecutas los 3 experimentos completos

---

## ⚡ Cambios importantes vs versión anterior

### ✅ Ahora SÍ tienes:
- ✔️ **Rutas base absolutas** (Windows con raw strings)
- ✔️ **Organización en sub-proyectos** (GCN, UMAP, JSON-NORM)
- ✔️ Nombres de gestos en lugar de números (0, 1, 2...)
- ✔️ Carpetas organizadas por tipo de experimento (G4-RESULTS-*)
- ✔️ Top-3 Accuracy incluido automáticamente
- ✔️ Formato estándar `Metric,Value` en metrics.csv
- ✔️ Configuración centralizada (un solo lugar para cambiar)
- ✔️ **Verificación automática de archivos generados**

### ❌ Ya NO necesitas:
- ✖️ Modificar múltiples variables en diferentes celdas
- ✖️ Buscar qué significa "Clase 0" o "Clase 15"
- ✖️ Crear manualmente las carpetas de salida
- ✖️ Cambiar rutas de guardado en cada celda
- ✖️ Recordar qué archivos deben generarse

---

## 🔄 Ejecutar los 3 experimentos completos

Si quieres ejecutar los 3 experimentos y generar la comparación:

**Opción A: Ejecutar todas las celdas (Recomendado)**
1. Seleccionar BASE_PATH (Opción 1, 2 o 3)
2. Ejecutar TODAS las celdas del notebook (Cell → Run All)
3. Resultado: 29 archivos totales (9 × 3 experimentos + 2 comparación)

**Opción B: Ejecutar manualmente uno por uno**

**Paso 1**: Cambiar a `'baseline'` y ejecutar celdas 1-17
```python
EXPERIMENT_TYPE = 'baseline'
```
→ Espera a que termine (verás los resultados en `[BASE_PATH]/G4-RESULTS-BASELINE/`)

**Paso 2**: Ejecutar celda #21 (Experimento 1: Class Weights)
→ Espera a que termine (resultados en `[BASE_PATH]/G4-RESULTS-CLASS-WEIGHTS/`)

**Paso 3**: Ejecutar celda #23 (Experimento 2: Label Smoothing)
→ Espera a que termine (resultados en `[BASE_PATH]/G4-RESULTS-LABEL-SMOOTHING/`)

**Paso 4**: Ejecutar celda #25 (Comparación)
→ Genera archivos de comparación en `[BASE_PATH]/`

**Paso 5**: Ejecutar última celda (Verificación)
→ Valida que los 29 archivos se hayan generado correctamente

---

## 📈 Ver resultados rápidamente

### Métricas principales:
```bash
cat G4-RESULTS-BASELINE/metrics.csv
```

### Visualizaciones:
Abre cualquiera de estos archivos PNG:
- `confusion_matrix.png` - Ver qué gestos se confunden (con nombres reales, no índices)
- `training_curves.png` - Ver cómo entrenó el modelo
- `per_class_analysis.png` - Ver rendimiento por gesto (con nombres reales)

---

## 🆘 Problemas comunes

### Error: "class_names not found"
**Solución**: Asegúrate de que existe el archivo:
```
./daataset/frame to frame/class_names.npy
```

### Error: "CUDA out of memory"
**Solución**: Reduce el batch_size en la configuración:
```python
config = {
    'batch_size': 4,  # Cambiar de 8 a 4
    ...
}
```

### Error: "Directory not found"
**Solución**: Las carpetas se crean automáticamente en formato G4. Si el error persiste, crea manualmente:
```bash
mkdir G4-RESULTS-BASELINE
mkdir G4-RESULTS-CLASS-WEIGHTS
mkdir G4-RESULTS-LABEL-SMOOTHING
```

---

## 💡 Tips

1. **Revisa las visualizaciones primero**: Son más fáciles de interpretar que los CSV
2. **Compara matrices de confusión**: Te dirá qué gestos son más difíciles
3. **Revisa el análisis por clase**: Identifica gestos problemáticos
4. **Guarda tus notebooks**: Si cambias hiperparámetros, guárdalo con otro nombre

---

## 📚 Más información

Para documentación completa, ver: `README_EXPERIMENTOS.md`

---

**¡Listo para empezar! 🎉**
