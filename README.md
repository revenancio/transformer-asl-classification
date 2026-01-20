# Transformer Encoder-Only para Clasificación de Video (ASL)

**Proyecto de Tesis:** Clasificación de secuencias temporales con Transformer Encoder-Only para American Sign Language (ASL).

## 📋 Descripción

Este proyecto implementa un modelo Transformer Encoder-Only para la clasificación de videos de lenguaje de señas americano (ASL). Se comparan diferentes configuraciones para mejorar el Macro-F1 y la generalización en datasets desbalanceados.

## 🏗️ Arquitectura

- **Modelo Base:** Transformer Encoder-Only (sin decoder)
- **Input:** 96 frames × 228 features (mano + pose + cara)
- **Clases:** 30 (A-Z, excluyendo J)
- **Framework:** PyTorch

## 🧪 Experimentos

### Exp 0 (G5) - Baseline
- Dropout: 0.1
- Sin class weights
- Sin label smoothing

### Exp 1 (G5.1) - Class Weights + Dropout 0.3
- Pesos de clase calculados por frecuencia inversa
- Dropout aumentado a 0.3
- Sin label smoothing

### Exp 2 (G5.2) - Dropout 0.3 + Label Smoothing
- Dropout: 0.3
- Label smoothing: 0.1
- Sin class weights

## 📊 Resultados

Los resultados detallados se encuentran en las carpetas `g5/`, `g5.1/`, `g5.2/` y en `experiments_comparison.csv`.

## 🚀 Uso

1. Instalar dependencias: `pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm`
2. Ejecutar el notebook `Experimento.ipynb` en Jupyter

## 📁 Estructura del Proyecto

```
.
├── Experimento.ipynb          # Notebook principal con código y análisis
├── g5/                        # Resultados baseline
├── g5.1/                      # Resultados experimento 1
├── g5.2/                      # Resultados experimento 2
├── experiments_comparison.*   # Comparación de experimentos
├── model_config_g5.json       # Configuración del modelo
├── .gitignore                 # Archivos ignorados
└── README.md                  # Este archivo
```

## 📈 Métricas Principales

- **Accuracy**
- **Macro-F1** (métrica principal para clases desbalanceadas)
- **Top-3 Accuracy**

## 🛠️ Tecnologías

- **PyTorch:** Framework de deep learning
- **Scikit-learn:** Métricas y preprocesamiento
- **Matplotlib/Seaborn:** Visualizaciones
- **Jupyter Notebook:** Entorno de desarrollo

## 📝 Licencia

Este proyecto es parte de una tesis académica. Contactar al autor para uso.

## 👤 Autor

[Tu Nombre] - Proyecto de tesis en [Universidad/Institution]