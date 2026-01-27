# 🔬 Guía de Experimentos - Transformer ASL Classification

## 📋 Resumen de Modificaciones Implementadas

Este documento describe las modificaciones realizadas al proyecto para mejorar la organización, trazabilidad y reproducibilidad de los experimentos de Machine Learning.

---

## ✨ Características Implementadas

### 1. 🧹 Limpieza Automática de Archivos Temporales
- Se eliminan automáticamente las carpetas `output_videos/`, `temp_results/`, `old_results/` al inicio de cada ejecución
- Asegura ejecuciones limpias sin conflictos de archivos antiguos

### 2. 🔧 Configuración Dinámica de Experimentos
Se agregó un sistema de configuración centralizado que permite cambiar fácilmente entre tres estrategias experimentales:

#### **Experimento: Baseline**
```python
EXPERIMENT_TYPE = 'baseline'
```
- **Directorio de salida**: `./results/exp_baseline`
- **Class Weights**: No
- **Label Smoothing**: 0.0
- **Dropout**: 0.1
- **Descripción**: Modelo base sin ajustes especiales

#### **Experimento: Class Weights**
```python
EXPERIMENT_TYPE = 'class_weights'
```
- **Directorio de salida**: `./results/exp_class_weights`
- **Class Weights**: Sí (balanceo de clases)
- **Label Smoothing**: 0.0
- **Dropout**: 0.3
- **Descripción**: Modelo con balanceo de clases por pesos

#### **Experimento: Label Smoothing**
```python
EXPERIMENT_TYPE = 'label_smoothing'
```
- **Directorio de salida**: `./results/exp_label_smoothing`
- **Class Weights**: No
- **Label Smoothing**: 0.1
- **Dropout**: 0.3
- **Descripción**: Modelo usando Label Smoothing

### 3. 📊 Visualizaciones con Etiquetas Legibles

**ANTES** (❌ Problemático):
- Matrices de confusión con índices numéricos (0, 1, 2, 3...)
- Imposible saber qué gesto representa cada número
- Difícil interpretación de resultados

**DESPUÉS** (✅ Mejorado):
- Todas las visualizaciones usan nombres reales de gestos
- Matriz de confusión legible con etiquetas en ambos ejes
- Análisis por clase con nombres descriptivos
- Colores diferenciados según rendimiento

### 4. 📁 Artefactos Generados por Experimento

Cada experimento genera automáticamente estos archivos en su directorio correspondiente:

#### **config.json**
Archivo JSON con todos los hiperparámetros y configuración del experimento:
```json
{
  "experiment_type": "baseline",
  "architecture": "TransformerEncoderOnly",
  "dropout": 0.1,
  "label_smoothing": 0.0,
  "use_class_weights": false,
  "test_accuracy": 0.9138,
  "test_macro_f1": 0.8736,
  "test_top3_accuracy": 0.9943,
  ...
}
```

#### **metrics.csv**
Formato estricto `Metric,Value` con métricas principales:
```csv
Metric,Value
Accuracy,0.9137931034482759
Macro-F1,0.8735598342661747
Top-3 Accuracy,0.9942528735632183
Test Loss,0.9224783046679064
```

#### **training_log.csv**
Historial completo del entrenamiento por época:
```csv
epoch,train_loss,train_acc,val_loss,val_acc,lr
0,2.5432,0.3456,2.1234,0.4123,0.0001
1,1.9876,0.5234,1.7654,0.5789,0.00009
...
```

#### **confusion_matrix.csv**
Matriz de confusión en formato CSV (valores numéricos)

#### **confusion_matrix.png**
Visualización de alta calidad (300 DPI) con:
- Nombres de clases en ejes X e Y
- Heatmap con valores anotados
- Colores profesionales
- Tamaño optimizado para publicaciones (20x18 inches)

#### **training_curves.png**
Gráficos de curvas de aprendizaje que incluyen:
- Loss de entrenamiento vs validación
- Accuracy de entrenamiento vs validación
- Programación del Learning Rate
- Métricas finales en Test Set
- Marca visual del mejor epoch

#### **per_class_analysis.png**
Análisis detallado por cada gesto con 3 gráficos:
- **Precision por clase**: Qué tan exacto es el modelo para cada gesto
- **Recall por clase**: Qué tan completo es el modelo (sensibilidad)
- **F1-Score por clase**: Balance entre precision y recall
- Código de colores: Verde (>0.7), Naranja (0.5-0.7), Rojo (<0.5)

#### **per_class_metrics.csv**
Tabla detallada con métricas individuales para cada clase

#### **best_model.pt**
Pesos del modelo correspondientes al mejor epoch de validación

---

## 🚀 Cómo Usar el Notebook Modificado

### Paso 1: Seleccionar Experimento
Edita la variable en la celda de configuración:
```python
EXPERIMENT_TYPE = 'baseline'  # Cambiar a: 'class_weights' o 'label_smoothing'
```

### Paso 2: Ejecutar Todo el Notebook
- Ejecuta todas las celdas secuencialmente
- El sistema automáticamente:
  - Limpia archivos temporales
  - Crea directorios de salida
  - Configura hiperparámetros
  - Entrena el modelo
  - Genera todas las visualizaciones
  - Guarda todos los artefactos

### Paso 3: Revisar Resultados
Navega a la carpeta correspondiente en `results/exp_[nombre]/` para encontrar todos los artefactos generados.

---

## 📂 Estructura de Directorios Final

```
transformer-asl-classification/
│
├── results/
│   ├── exp_baseline/
│   │   ├── config.json
│   │   ├── metrics.csv
│   │   ├── training_log.csv
│   │   ├── confusion_matrix.csv
│   │   ├── confusion_matrix.png
│   │   ├── training_curves.png
│   │   ├── per_class_analysis.png
│   │   ├── per_class_metrics.csv
│   │   └── best_model.pt
│   │
│   ├── exp_class_weights/
│   │   └── [mismos archivos]
│   │
│   └── exp_label_smoothing/
│       └── [mismos archivos]
│
├── embedding_frame_gcn/          # (Reorganizado desde G4-EMBEDDING FRAME A FRAME GCN)
│   ├── results_baseline/
│   ├── results_class_weights/
│   └── results_label_smoothing/
│
├── embedding_frame_umap/         # (Reorganizado desde G4-EMBEDDING FRAME A FRAME UMAP)
│   ├── results_baseline/
│   ├── results_class_weights/
│   └── results_label_smoothing/
│
├── json_normalized/               # (Reorganizado desde G4-JSON-NORM)
│   ├── results_baseline/
│   ├── results_class_weights/
│   └── results_label_smoothing/
│
├── Experimento.ipynb             # ⚡ Notebook principal (MODIFICADO)
├── reorganize_and_visualize.py   # 🆕 Script de reorganización
└── README_EXPERIMENTOS.md        # 🆕 Esta guía
```

---

## 📊 Comparación de Experimentos

Una vez ejecutados los tres experimentos, puedes comparar fácilmente los resultados:

| Métrica | Baseline | Class Weights | Label Smoothing |
|---------|----------|---------------|-----------------|
| Accuracy | ? | ? | ? |
| Macro-F1 | ? | ? | ? |
| Top-3 Accuracy | ? | ? | ? |
| Test Loss | ? | ? | ? |

*(Completar después de ejecutar los experimentos)*

---

## 🔍 Mejoras Clave

### ✅ Antes vs Después

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Organización** | Carpetas con nombres inconsistentes (G4, G5, g8.0...) | Nomenclatura clara y jerárquica |
| **Visualizaciones** | Índices numéricos (0, 1, 2...) | Nombres reales de gestos |
| **Configuración** | Hard-coded en múltiples lugares | Centralizada y dinámica |
| **Reproducibilidad** | Difícil cambiar entre experimentos | Un solo cambio de variable |
| **Formato de Métricas** | Inconsistente | Formato estándar Metric,Value |
| **Documentación** | Mínima o ausente | Config.json completo por experimento |

---

## 💡 Consejos para Mejores Resultados

1. **Ejecuta experimentos en orden**: Baseline → Class Weights → Label Smoothing
2. **Guarda checkpoints frecuentemente**: El early stopping ya lo hace automáticamente
3. **Compara resultados visuales**: Las imágenes PNG son más fáciles de interpretar que los CSV
4. **Revisa el análisis por clase**: Identifica qué gestos necesitan más datos o mejoras
5. **Documenta cambios**: Si modificas hiperparámetros manualmente, anótalos en config.json

---

## 🛠️ Herramientas Adicionales

### Script de Reorganización
El archivo `reorganize_and_visualize.py` puede usarse para reorganizar experimentos antiguos:

```bash
python reorganize_and_visualize.py
```

Este script:
- Carga nombres de clases desde `daataset/frame to frame/class_names.npy`
- Reorganiza carpetas antiguas (G4, G5, etc.)
- Genera visualizaciones con nombres de clases
- Crea reportes comparativos

---

## 📞 Soporte

Si encuentras problemas o necesitas modificar la configuración:

1. Revisa que `class_names` esté correctamente cargado
2. Verifica que las rutas de archivos sean correctas
3. Asegúrate de tener instaladas todas las dependencias: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `torch`

---

## 📝 Changelog

### v2.0 - 2026-01-26
- ✅ Agregada limpieza automática de archivos temporales
- ✅ Implementado sistema de configuración dinámica
- ✅ Visualizaciones con nombres reales de clases
- ✅ Generación automática de todos los artefactos
- ✅ Formato estandarizado de métricas (Metric,Value)
- ✅ Inclusión de Top-3 Accuracy
- ✅ Análisis detallado por clase con nombres
- ✅ Documentación completa
- ✅ Reestructuración de directorios con nomenclatura consistente

---

**Autor**: MLOps Engineer  
**Fecha**: 26 de Enero, 2026  
**Versión**: 2.0
