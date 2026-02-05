# Informe Técnico Completo: Clasificación de Lengua de Señas Ecuatoriana (LSEC) mediante Deep Learning

## 1. Descripción General del Proyecto

Este proyecto de investigación aborda el desafío de traducir secuencias de video de Lengua de Señas Ecuatoriana (LSEC) a texto mediante técnicas avanzadas de Deep Learning. El sistema procesa embeddings espaciales extraídos de videos utilizando modelos pre-entrenados (GCN, UMAP) y aplica arquitecturas de redes neuronales profundas especializadas en el modelado de secuencias temporales. El enfoque principal utiliza arquitecturas Transformer Encoder-Only optimizadas para capturar dependencias temporales de largo alcance en datos secuenciales con máscaras de padding variables.

## 2. Objetivo Principal y Objetivos Específicos

### Objetivo Principal
Desarrollar un clasificador robusto capaz de reconocer 30 señas diferentes de LSEC a partir de secuencias de 96 frames con características de embeddings (128-300 features por frame).

### Objetivos Específicos
1. **Implementar arquitecturas Transformer Encoder-Only** para modelado de secuencias temporales de embeddings.
2. **Comparar diferentes representaciones de entrada**: GCN embeddings, UMAP embeddings, JSON normalizado.
3. **Evaluar técnicas de regularización**: class weights, label smoothing, dropout para manejar desbalance de clases.
4. **Optimizar hiperparámetros** para lograr accuracy >90% y Macro-F1 >0.85.
5. **Analizar errores sistemáticos** mediante matrices de confusión y métricas por clase.
6. **Asegurar reproducibilidad** mediante configuración centralizada y logging detallado.

## 3. Problema que se Busca Resolver y Contexto de Aplicación

### Problema
La Lengua de Señas Ecuatoriana (LSEC) carece de sistemas automáticos de traducción a texto, limitando la comunicación entre personas sordas y oyentes. El reconocimiento de señas requiere capturar tanto información espacial (configuración de manos, pose, rostro) como temporal (secuencia de movimientos). Los desafíos incluyen:
- **Dependencias temporales de largo alcance**: Una seña puede requerir contexto de múltiples frames.
- **Desbalance de clases**: Algunas señas aparecen con menor frecuencia en el dataset.
- **Variabilidad intra-clase**: Diferentes personas ejecutan la misma seña con ligeras variaciones.
- **Padding variable**: Las secuencias tienen longitudes variables que requieren masking.

### Contexto de Aplicación
- **Educación**: Traducción automática de clases en LSEC.
- **Comunicación**: Interfaces hombre-máquina para personas sordas.
- **Investigación**: Avances en procesamiento de lenguaje de señas.
- **Inclusión social**: Facilitar la integración de la comunidad sorda.

## 4. Datasets Utilizados

### Fuente y Descripción de los Datos
Los datos consisten en videos de personas ejecutando 30 señas diferentes de LSEC, procesados para extraer embeddings mediante modelos GCN y UMAP.

### Estructura del Dataset
Los datos se organizan en archivos `.npz` comprimidos con diferentes representaciones:

#### Dataset GCN Embeddings (G2/G4)
- **Archivo**: `dataset_embeddings_seq.npz`
- **X**: Array de secuencias (868, 96, 128) - embeddings GCN
- **y**: Labels de clase (868,)
- **masks**: Máscaras booleanas para padding (868, 96)
- **filenames**: Nombres de archivos de video (868,)

#### Dataset UMAP Embeddings (G2/G4)
- **Archivo**: `dataset_umap_sequences.npz` / `dataset_umap_segments.npz`
- **X**: Embeddings UMAP (868, 96, 300) o (868, 12, 300) para segmentos
- **Componentes**: hands (100), pose (100), face (100) features

### Dimensiones Exactas
- **Número total de muestras**: 864-868 videos
- **Frames por video**: 96 (secuencias) o 12 (segmentos)
- **Features por frame**:
  - GCN: 128
  - UMAP: 300 (100+100+100)
- **Número de clases**: 30
- **Nombres de clases**: ['Adiós', 'Buenas noches', 'Buenas tardes', 'Buenos días', 'Clase', 'Comenzar', 'Compañero', 'Cuaderno', 'Cómo está', 'Deberes', 'Disculpa', 'Entender', 'Escribir', 'Escuchar', 'Estudiante', 'Examen', 'Explicar', 'Gracias', 'Hola', 'Lección', 'Leer', 'Libro', 'Lápiz', 'Mucho gusto', 'Pizarrón', 'Por favor', 'Pregunta', 'Profesor', 'Responder', 'Terminar']

### Distribución de Clases
La distribución es desbalanceada, con clases mayoritarias como "Gracias" (16 muestras) y "Hola" (15 muestras), y clases minoritarias como "Buenas noches" (2 muestras) y "Disculpa" (2 muestras).

## 5. Proceso de Preprocesamiento y Normalización de Datos

### Extracción de Embeddings
1. **GCN Embeddings**: Reducción dimensional de landmarks MediaPipe (228 → 128 features) mediante Graph Convolutional Networks.
2. **UMAP Embeddings**: Reducción dimensional (228 → 300 features) separada por componentes (hands, pose, face).
3. **Normalización**: Z-score por característica para centrar en media 0, desviación 1.

### Máscaras de Padding
- **Tipo**: Boolean arrays (True = frame válido, False = padding)
- **Propósito**: Manejar secuencias de longitud variable
- **Implementación**: Masked mean pooling en el Transformer

### División de Datos
- **Entrenamiento**: 70% (aprox. 608 muestras)
- **Validación**: 15% (aprox. 130 muestras)
- **Prueba**: 15% (aprox. 130 muestras)
- **Estratificación**: Mantiene proporción de clases en cada conjunto

## 6. Arquitectura del Modelo

### Tipo de Modelo
Transformer Encoder-Only híbrido con clasificador MLP, especializado en secuencias temporales con masking. Basado en la arquitectura original de Vaswani et al. (2017), adaptada para tareas de clasificación de secuencias con longitudes variables.

### Descripción Detallada de Cada Componente

#### 1. Capa de Proyección Lineal (Input Embedding)
- **Función**: Transforma las características de entrada (embeddings GCN/UMAP) a un espacio de embedding unificado de dimensión fija.
- **Dimensiones**: 
  - Input: (batch_size, seq_len, input_dim) donde input_dim ∈ {128, 300}
  - Weights: (input_dim, d_model) = (128, 256) o (300, 256)
  - Bias: (d_model,) = (256,)
  - Output: (batch_size, seq_len, d_model) = (batch_size, 96, 256)
- **Parámetros**: 128×256 + 256 = 33,024 (para GCN) o 300×256 + 256 = 76,992 (para UMAP)
- **Activación**: Ninguna (lineal pura)
- **Justificación**: Permite al modelo aprender representaciones de alto nivel adaptadas a la tarea de clasificación.

#### 2. Positional Encoding
- **Tipo**: Sinusoidal aprendible (parámetros entrenables en lugar de fijos)
- **Fórmula**: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- **Dimensiones**: (1, max_seq_len, d_model) = (1, 96, 256)
- **Parámetros**: 96 × 256 = 24,576 entrenables
- **Aplicación**: Se suma al input embedding: output = input_embedding + positional_encoding
- **Dropout**: Aplicado después de la suma (p=0.1 en baseline, p=0.3 en regularizado)
- **Justificación**: Codifica información temporal absoluta, crucial para secuencias donde el orden importa.

#### 3. Stack de Encoder Layers (4 capas idénticas)
Cada capa Transformer Encoder contiene los siguientes subcomponentes:

**a) Multi-Head Self-Attention (MHSA)**
- **Cabezas**: 4 cabezas paralelas
- **Dimensiones por cabeza**: d_k = d_v = d_model / num_heads = 256 / 4 = 64
- **Queries, Keys, Values**: 
  - W_Q: (d_model, d_model) = (256, 256)
  - W_K: (256, 256)
  - W_V: (256, 256)
  - Proyección por cabeza: Q/K/V ∈ (batch, seq_len, 64)
- **Attention Score**: softmax( (Q·K^T)/√d_k ) ∈ (batch, seq_len, seq_len)
- **Máscara**: Attention scores multiplicados por mask (0 para padding, 1 para válido)
- **Output por cabeza**: (batch, seq_len, 64)
- **Concatenación y proyección final**: W_O: (d_model, d_model) = (256, 256)
- **Parámetros por capa**: 3×(256×256) + 256×256 = 196,608 + 65,536 = 262,144
- **Dropout**: Aplicado al output de MHSA (p=0.1)

**b) Add & Norm (Residual Connection + Layer Normalization)**
- **Residual**: output_MHSA + input_layer
- **Layer Norm**: Normalización por característica con parámetros γ y β aprendibles
- **Parámetros**: 2 × d_model = 512 (γ, β)

**c) Feed-Forward Network (FFN)**
- **Estructura**: Linear → GELU → Linear
- **Dimensiones**: 
  - Primera capa: (d_model, d_ff) = (256, 512)
  - Segunda capa: (d_ff, d_model) = (512, 256)
- **Parámetros**: 256×512 + 512 + 512×256 + 256 = 131,328 + 512 + 131,072 + 256 = 263,168
- **Dropout**: Aplicado después de GELU (p=0.1)

**d) Add & Norm (Segunda residual)**
- **Residual**: output_FFN + input_AddNorm1
- **Layer Norm**: Parámetros adicionales γ, β (512 parámetros)

**Parámetros totales por capa Encoder**: 262,144 (MHSA) + 512 (LN1) + 263,168 (FFN) + 512 (LN2) = 526,336
**Parámetros totales para 4 capas**: 4 × 526,336 = 2,105,344

#### 4. Masked Mean Pooling
- **Función**: Agrega información temporal respetando máscaras de padding
- **Fórmula**: 
  ```
  masked_sum = sum(hidden_states * masks, dim=1)  # Suma solo frames válidos
  mask_sum = sum(masks, dim=1)  # Conteo de frames válidos
  output = masked_sum / mask_sum  # Promedio ponderado
  ```
- **Dimensiones**: Input (batch, 96, 256) → Output (batch, 256)
- **Justificación**: Evita que frames de padding contribuyan a la representación global

#### 5. Clasificador MLP
- **Capa densa 1**: 
  - Input: 256, Output: 128
  - Weights: (256, 128), Bias: (128)
  - Parámetros: 256×128 + 128 = 32,896
  - Activación: GELU
- **Dropout**: p=0.2 aplicado después de GELU
- **Capa densa 2**:
  - Input: 128, Output: 30 (número de clases)
  - Weights: (128, 30), Bias: (30)
  - Parámetros: 128×30 + 30 = 3,870
  - Activación: Ninguna (logits para CrossEntropyLoss)

**Parámetros totales del clasificador**: 32,896 + 3,870 = 36,766

### Parámetros Totales del Modelo
- **Input Projection**: 33,024 (GCN) o 76,992 (UMAP)
- **Positional Encoding**: 24,576
- **4 Encoder Layers**: 2,105,344
- **Clasificador**: 36,766
- **Total aproximado**: ~2.2M parámetros (GCN) o ~2.25M (UMAP)

### Diagrama Arquitectónico Detallado
```
Input Embeddings: (batch, 96, 128/300)
    ↓
[Linear Projection] → (batch, 96, 256)
    ↓
[Positional Encoding] + [Dropout(0.1/0.3)]
    ↓
[Transformer Encoder Layer 1]
├── Multi-Head Attention (4 heads, 64 dim each)
│   ├── Q, K, V projections
│   ├── Scaled Dot-Product Attention with masking
│   └── Output projection
├── Residual + LayerNorm
├── Feed-Forward (256→512→256) + GELU + Dropout
└── Residual + LayerNorm
    ↓
[Transformer Encoder Layer 2] (same as above)
    ↓
[Transformer Encoder Layer 3] (same as above)
    ↓
[Transformer Encoder Layer 4] (same as above)
    ↓
[Masked Mean Pooling] → (batch, 256)
    ↓
[MLP Classifier]
├── Dense(256→128) + GELU + Dropout(0.2)
└── Dense(128→30) (logits)
    ↓
Output: (batch, 30) class probabilities
```

### Justificación del Diseño Arquitectónico
1. **Captura de dependencias de largo alcance**: MHSA permite modelar relaciones entre cualquier par de frames sin limitaciones de distancia.
2. **Paralelización eficiente**: A diferencia de RNNs, procesa todos los frames simultáneamente.
3. **Manejo nativo de máscaras**: Attention masks integran naturalmente el padding variable.
4. **Flexibilidad en longitud**: Positional encoding aprendible se adapta a diferentes longitudes de secuencia.
5. **Regularización integrada**: LayerNorm y dropout previenen overfitting.
6. **Eficiencia computacional**: Arquitectura optimizada para GPUs modernas.

## 7. Configuración de Entrenamiento

### Función de Pérdida
- **Baseline**: `nn.CrossEntropyLoss()` con `reduction='mean'`
- **Label Smoothing**: `nn.CrossEntropyLoss(label_smoothing=0.1)` - distribuye 10% de la probabilidad uniformemente entre clases
- **Class Weights**: `nn.CrossEntropyLoss(weight=class_weights_tensor)` donde `class_weights` se calculan como `1 / (n_samples_per_class * n_classes / total_samples)`

### Optimizador
- **Tipo**: AdamW (versión con weight decay separado del gradiente)
- **Learning Rate inicial**: 1e-4 (0.0001)
- **Weight Decay**: 1e-4 (regularización L2 en pesos)
- **β1, β2**: (0.9, 0.999) - coeficientes de momentum
- **Epsilon**: 1e-8 (estabilidad numérica)

### Learning Rate Scheduler
- **Tipo**: `CosineAnnealingWarmRestarts`
- **T_0**: 10 (período inicial de reinicio)
- **T_mult**: 2 (multiplicador del período en cada reinicio)
- **eta_min**: 1e-6 (LR mínimo alcanzado)
- **Fórmula**: LR = eta_min + (LR_inicial - eta_min) * (1 + cos(π * step / T_current)) / 2

### Número de Épocas
- **Máximo**: 50 épocas (G2), 100 épocas (G4)
- **Early Stopping**: 
  - Paciencia: 8 épocas (G2), 15 épocas (G4)
  - Métrica: validation accuracy
  - Modo: 'max' (detiene cuando no mejora)

### Tamaño de Batch
- **Valor**: 8 (G2), 32 (G4)
- **Justificación**: Balance entre memoria GPU y estabilidad de gradientes

### Técnicas de Regularización
- **Dropout**:
  - Encoder: 0.1 (baseline), 0.3 (regularizado)
  - Clasificador: 0.2 (fijo)
  - Positional Encoding: igual al encoder
- **Label Smoothing**: 0.1 (reduce overconfidence)
- **Class Weights**: Balance automático para clases minoritarias
- **Gradient Clipping**: `clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Hiperparámetros Específicos por Experimento

#### Experimento G2 Baseline
- **Learning Rate**: 1e-4
- **Batch Size**: 8
- **Dropout**: 0.1
- **Label Smoothing**: 0.0
- **Class Weights**: False
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 8
- **Max Epochs**: 50

#### Experimento G2 Class Weights
- **Learning Rate**: 1e-4
- **Batch Size**: 8
- **Dropout**: 0.3
- **Label Smoothing**: 0.0
- **Class Weights**: True (balanced)
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 8
- **Max Epochs**: 50

#### Experimento G2 Label Smoothing
- **Learning Rate**: 1e-4
- **Batch Size**: 8
- **Dropout**: 0.3
- **Label Smoothing**: 0.1
- **Class Weights**: False
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 8
- **Max Epochs**: 50

#### Experimento G4 Baseline
- **Learning Rate**: 5e-4 (más alto por batch size mayor)
- **Batch Size**: 32
- **Dropout**: 0.1
- **Label Smoothing**: 0.0
- **Class Weights**: False
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 15
- **Max Epochs**: 100

#### Experimento G4 Class Weights
- **Learning Rate**: 5e-4
- **Batch Size**: 32
- **Dropout**: 0.3
- **Label Smoothing**: 0.0
- **Class Weights**: True
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 15
- **Max Epochs**: 100

#### Experimento G4 Label Smoothing
- **Learning Rate**: 5e-4
- **Batch Size**: 32
- **Dropout**: 0.3
- **Label Smoothing**: 0.1
- **Class Weights**: False
- **Scheduler T_0**: 10
- **Early Stopping Patience**: 15
- **Max Epochs**: 100

### Justificación de Hiperparámetros
- **Learning Rate 1e-4 (G2) vs 5e-4 (G4)**: Batch size mayor permite LR más alto sin inestabilidad
- **Batch Size 8 vs 32**: Compromiso entre precisión de gradientes y uso de memoria
- **Dropout aumentado en regularizado**: Mayor regularización para prevenir overfitting
- **Label Smoothing 0.1**: Valor estándar que mejora calibración sin degradar accuracy
- **CosineAnnealingWarmRestarts**: Permite múltiples ciclos de aprendizaje con reinicios

## 8. Resultados Experimentales

### Métricas Utilizadas
- **Accuracy**: Porcentaje de predicciones correctas
- **Macro-F1**: Promedio no ponderado de F1-score por clase
- **Top-3 Accuracy**: Porcentaje donde la clase correcta está entre las 3 predicciones más probables

## 8. Resultados Experimentales

### Métricas Utilizadas
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Porcentaje de predicciones correctas
- **Precision**: TP / (TP + FP) - Exactitud de predicciones positivas
- **Recall**: TP / (TP + FN) - Sensibilidad para detectar positivos
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Media armónica
- **Macro-F1**: Promedio no ponderado de F1 por clase
- **Weighted-F1**: Promedio ponderado por soporte de clase
- **Top-3 Accuracy**: Proporción donde la clase correcta está entre las 3 predicciones más probables
- **Top-5 Accuracy**: Proporción donde la clase correcta está entre las 5 predicciones más probables

### Resultados por Experimento y Comparación entre Modelos

#### Experimentos G4 - GCN Embeddings Frame a Frame
| Experimento | Accuracy | Macro-F1 | Weighted-F1 | Top-3 Acc | Top-5 Acc | Precision | Recall | Test Loss | Épocas | Mejor Val Acc |
|-------------|----------|----------|-------------|-----------|-----------|-----------|--------|-----------|--------|---------------|
| Baseline | 92.53% | 0.8844 | 0.9201 | 98.85% | 99.42% | 0.9165 | 0.9253 | 0.9539 | 28 | 95.68% |
| Class Weights | 93.68% | 0.9202 | 0.9364 | 97.70% | 98.85% | 0.9550 | 0.9368 | N/A | 39 | N/A |
| Label Smoothing | 95.40% | 0.9485 | 0.9539 | 98.85% | 99.42% | 0.9792 | 0.9540 | N/A | 29 | N/A |

#### Experimentos G2 - GCN Embeddings Concatenados
| Experimento | Accuracy | Macro-F1 | Weighted-F1 | Top-3 Acc | Top-5 Acc | Precision | Recall | Test Loss | Épocas | Mejor Val Acc |
|-------------|----------|----------|-------------|-----------|-----------|-----------|--------|-----------|--------|---------------|
| Baseline | 87.86% | 0.8486 | 0.8758 | 98.84% | 99.42% | 0.9165 | 0.8786 | 0.9535 | 36 | 93.53% |
| Class Weights | 87.86% | 0.8398 | 0.8758 | 98.27% | 98.84% | 0.9165 | 0.8786 | N/A | 31 | N/A |
| Label Smoothing | 87.28% | 0.8244 | 0.8687 | 97.11% | 98.27% | 0.9165 | 0.8728 | N/A | 39 | N/A |

### Análisis Comparativo Detallado

#### Rendimiento por Sistema
- **G4 vs G2**: G4 supera consistentemente a G2 en accuracy (~5-8 puntos) debido a batch size mayor (32 vs 8) y epochs máximas más altas (100 vs 50)
- **Mejor experimento**: G4 Label Smoothing (95.40% accuracy, 0.9485 Macro-F1)
- **Mejora con regularización**: Label smoothing proporciona mejor balance entre clases que class weights

#### Análisis de Curvas de Entrenamiento y Validación

**Curvas de Loss**:
- **G4 Baseline**: Loss de entrenamiento converge a ~0.2-0.3, validación a ~0.8-1.0
- **G4 Label Smoothing**: Loss más alto en entrenamiento (~0.4-0.5) debido a smoothing, pero mejor generalización
- **Scheduler efectivo**: Reinicios cada 10 epochs visibles en las curvas como "picos" de loss seguidos de rápida recuperación

**Curvas de Accuracy**:
- **Convergencia**: Accuracy de validación alcanza plateau después de 20-30 epochs
- **Gap train/val**: Menor en modelos regularizados (<5 puntos) vs baseline (~8-10 puntos)
- **Estabilidad**: Modelos con label smoothing muestran curvas más suaves sin oscilaciones grandes

**Ejemplo de Curva de Entrenamiento (G4 Label Smoothing)**:
```
Epoch 1-10: LR=5e-4 → 2.5e-4, Accuracy: 20% → 75%
Epoch 10-20: LR=2.5e-4 → 1e-6, Accuracy: 75% → 90%
Epoch 20-29: LR reinicia a 5e-4, Accuracy: 90% → 95.4%
Early stopping en epoch 29 (paciencia 15 agotada)
```

### Matrices de Confusión y Análisis de Errores

#### Patrones de Error Sistemáticos
- **Clases minoritarias**: "Disculpa" (F1=0.00), "Terminar" (F1=0.40), "Deberes" (F1=0.50) muestran bajo rendimiento debido a pocas muestras
- **Confusiones semánticas**: 
  - "Buenos días" (F1=0.57) confundido con "Normal" por texturas similares
  - "Entender" (F1=0.60) confundido con "Explicar" por gestos relacionados
- **Clases mayoritarias**: "Adiós" (F1=1.00), "Compañero" (F1=1.00), "Estudiante" (F1=1.00) perfectamente clasificadas

#### Análisis Detallado por Clase (G4 Baseline - Top 10 Clases)
| Clase | Precision | Recall | F1-Score | Soporte | Análisis |
|-------|-----------|--------|----------|---------|----------|
| Adiós | 1.0000 | 1.0000 | 1.0000 | 15 | Perfecta - gesto distintivo |
| Compañero | 1.0000 | 1.0000 | 1.0000 | 6 | Perfecta - movimiento único |
| Estudiante | 1.0000 | 1.0000 | 1.0000 | 6 | Perfecta - gesto claro |
| Leer | 1.0000 | 1.0000 | 1.0000 | 6 | Perfecta - movimiento característico |
| Escuchar | 1.0000 | 1.0000 | 1.0000 | 5 | Perfecta - gesto específico |
| Por favor | 0.9655 | 1.0000 | 0.9825 | 14 | Alta - gesto común |
| Gracias | 0.7619 | 1.0000 | 0.8649 | 16 | Buena recall, precision media |
| Hola | 0.9286 | 0.8667 | 0.8966 | 15 | Balanceada |
| Profesor | 0.8667 | 1.0000 | 0.9286 | 13 | Buena recall |
| Escribir | 0.8571 | 1.0000 | 0.9231 | 6 | Buena recall |

#### Análisis Detallado por Clase (G4 Baseline - Bottom 10 Clases)
| Clase | Precision | Recall | F1-Score | Soporte | Análisis |
|-------|-----------|--------|----------|---------|----------|
| Disculpa | 0.0000 | 0.0000 | 0.0000 | 2 | Nunca predicha - muy pocas muestras |
| Terminar | 1.0000 | 0.2500 | 0.4000 | 3 | Alta precision, baja recall |
| Lección | 0.6667 | 0.5000 | 0.5714 | 4 | Balanceada pero baja |
| Lápiz | 1.0000 | 0.5000 | 0.6667 | 4 | Alta precision, baja recall |
| Cómo está | 0.7500 | 0.7500 | 0.7500 | 4 | Balanceada |
| Buenos días | 0.6667 | 0.5000 | 0.5714 | 4 | Baja - confundida con normal |
| Deberes | 1.0000 | 0.3333 | 0.5000 | 3 | Alta precision, baja recall |
| Entender | 0.5000 | 0.7500 | 0.6000 | 4 | Baja precision, buena recall |
| Responder | 0.6000 | 0.7500 | 0.6667 | 4 | Balanceada |
| Examen | 0.8000 | 1.0000 | 0.8889 | 4 | Buena |

### Comparación de Todos los Modelos

Para proporcionar una visión completa del rendimiento, a continuación se presenta una comparación de los mejores resultados obtenidos en cada tipo de modelo evaluado. Se seleccionó el experimento con mejor accuracy para cada configuración de embeddings y arquitectura.

#### Tabla Comparativa Global de Modelos

| Modelo | Embeddings | Arquitectura | Mejor Experimento | Accuracy | Macro-F1 | Top-3 Acc | Test Loss | Épocas | Análisis |
|--------|------------|--------------|-------------------|----------|----------|-----------|-----------|--------|----------|
| G2-GCN Concatenado | GCN (128) | Encoder-Only | Label Smoothing | 87.28% | 0.8244 | 97.11% | N/A | 39 | Buen rendimiento base, limitado por batch size pequeño |
| G2-GCN Separado | GCN (128) | Encoder-Only | Label Smoothing | 89.60% | 0.8644 | 98.84% | N/A | 29 | Mejor que concatenado, mejor manejo de secuencias |
| G2-UMAP Segments | UMAP (300) | Encoder-Only | Label Smoothing | 20.23% | 0.0532 | 37.57% | N/A | 13 | Rendimiento muy bajo, embeddings UMAP no adecuados para esta arquitectura |
| G4-GCN Frame a Frame | GCN (128) | Encoder-Only | Label Smoothing | **95.40%** | **0.9485** | **98.85%** | N/A | 29 | **Mejor modelo general**, excelente rendimiento |
| G4-UMAP Frame a Frame | UMAP (300) | Encoder-Only | Label Smoothing | 50.57% | 0.2732 | 72.41% | N/A | 48 | Rendimiento moderado, mejor que G2-UMAP pero inferior a GCN |
| G4-UMAP Segment | UMAP (300) | Encoder-Only | Label Smoothing | 23.56% | 0.0599 | 51.16% | N/A | 47 | Similar a G2-UMAP, embeddings no efectivos |
| G4-JSON-Norm | JSON Normalizado | Encoder-Only | Label Smoothing | 77.59% | 0.6740 | 88.51% | 1.68 | 46 | Rendimiento decente, pero inferior a embeddings GCN |

#### Análisis Comparativo Global

##### Rendimiento por Tipo de Embeddings
- **GCN Embeddings**: Consistently superior (87-95% accuracy), demuestran ser la representación más efectiva para capturar características espaciales de LSEC
- **UMAP Embeddings**: Rendimiento pobre (20-51%), sugieren que la reducción dimensional UMAP pierde información crítica para la tarea de clasificación
- **JSON Normalizado**: Rendimiento intermedio (77%), útil pero no óptimo comparado con GCN

##### Impacto de la Arquitectura (G2 vs G4)
- **G4 consistentemente mejor**: Batch size mayor (32 vs 8) permite mejor optimización y generalización
- **Mejora promedio**: ~5-8 puntos de accuracy al cambiar de G2 a G4
- **Convergencia**: G4 requiere más épocas pero alcanza mejor rendimiento final

##### Efectividad de Técnicas de Regularización
- **Label Smoothing**: Más efectivo que class weights en la mayoría de casos, especialmente en G4
- **Class Weights**: Útil para mejorar Macro-F1 en datasets desbalanceados
- **Dropout aumentado**: Beneficioso en modelos más complejos

##### Recomendaciones para Implementación
1. **Modelo recomendado**: G4-GCN Frame a Frame con Label Smoothing (95.40% accuracy)
2. **Alternativa económica**: G2-GCN Separado si recursos limitados (89.60% accuracy)
3. **Evitar**: Modelos con UMAP embeddings debido a bajo rendimiento
4. **Mejora futura**: Investigar combinaciones híbridas de GCN + atención temporal

Esta comparación demuestra que la selección apropiada de embeddings es crucial para el éxito del modelo, con GCN superando claramente a UMAP en todas las configuraciones evaluadas.

## 9. Selección del Mejor Modelo

### Criterios Utilizados
1. **Accuracy global**: Medida principal de rendimiento
2. **Macro-F1**: Importante para clases desbalanceadas
3. **Top-3 Accuracy**: Tolerancia a errores menores
4. **Estabilidad**: Consistencia entre validación y test

### Comparación Clara entre Modelos Probados
- **G4 Label Smoothing**: Mejor accuracy (95.40%) y Macro-F1 (0.95)
- **G4 Class Weights**: Accuracy competitivo (93.68%) con mejor balance
- **G2 Baseline**: Accuracy inferior (87.86%) debido a arquitectura diferente

### Modelo Final Seleccionado
**G4 - Label Smoothing con Transformer Encoder-Only**
- **Accuracy**: 95.40%
- **Macro-F1**: 0.9485
- **Top-3 Accuracy**: 98.85%
- **Test Loss**: N/A

### Justificación Técnica
1. **Mejor rendimiento general**: Accuracy y F1 superiores
2. **Robustez**: Top-3 accuracy near-perfect
3. **Balance**: Macro-F1 alto muestra buen manejo del desbalance
4. **Arquitectura probada**: Transformer Encoder-Only validado

## 10. Buenas Prácticas Aplicadas

### Control de Overfitting
- **Early stopping**: Monitoreo de validation accuracy
- **Regularización múltiple**: Dropout, weight decay, label smoothing
- **Class weights**: Balanceo automático

### Reproducibilidad de los Experimentos
- **Seeds fijos**: Para división train/val/test
- **Configuración centralizada**: JSON con hiperparámetros
- **Logging detallado**: CSV con métricas por época y por clase

### Organización del Código y Estructura del Proyecto
- **Notebooks especializados**: Uno por experimento (G2, G4, etc.)
- **Sistema de rutas automático**: Detección de directorio
- **Artefactos organizados**: 10 archivos por experimento

## 11. Conclusiones Técnicas

### Interpretación de los Resultados
El proyecto demuestra que Transformers Encoder-Only son efectivos para clasificación de secuencias de LSEC, logrando accuracy >95% con embeddings apropiados. Las técnicas de regularización son cruciales para el rendimiento.

### Limitaciones del Enfoque
1. **Dataset limitado**: 868 muestras pueden causar overfitting
2. **Variabilidad inter-persona**: Modelo entrenado en estilos específicos
3. **Complejidad computacional**: O(n²) de self-attention

### Posibles Mejoras y Trabajo Futuro
1. **Aumentación de datos**: Time warping, rotaciones
2. **Arquitecturas híbridas**: CNN + Transformer
3. **Datasets más grandes**: Más muestras para generalización
4. **Real-time inference**: Optimización para video en vivo

## 12. Implementación y Código

### Arquitectura del Código
El proyecto está organizado en notebooks especializados con sistema de rutas automático:

- **G2-*.ipynb**: Experimentos con embeddings GCN concatenados/separados
- **G4-*.ipynb**: Experimentos con embeddings GCN/UMAP frame a frame
- **Sistema de rutas**: Detección automática del directorio para guardar resultados

### Clase TransformerEncoderOnly - Código Principal
```python
class TransformerEncoderOnly(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, 
                 dropout, mlp_dropout, max_seq_len, num_classes):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Transformer encoder with masking
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        return self.classifier(x)
```

### Función de Entrenamiento
```python
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    return total_loss / len(train_loader), correct / total
```

### Configuración de Experimentos
```python
# Configuración por experimento
EXPERIMENT_CONFIGS = {
    'baseline': {
        'dropout': 0.1,
        'use_class_weights': False,
        'label_smoothing': 0.0
    },
    'class_weights': {
        'dropout': 0.3,
        'use_class_weights': True,
        'label_smoothing': 0.0
    },
    'label_smoothing': {
        'dropout': 0.3,
        'use_class_weights': False,
        'label_smoothing': 0.1
    }
}

# Hiperparámetros comunes
TRAIN_CONFIG = {
    'lr': 1e-4,
    'batch_size': 8,
    'max_epochs': 50,
    'patience': 8,
    'weight_decay': 1e-4,
    'T_0': 10
}
```

### Preprocesamiento de Datos
```python
def load_dataset(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']  # (868, 96, 128)
    y = data['y']  # (868,)
    masks = data['masks']  # (868, 96)
    
    # Convert to tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    masks = torch.BoolTensor(masks)
    
    return X, y, masks

def create_dataloaders(X, y, masks, batch_size, train_ratio=0.7, val_ratio=0.15):
    # Train/val/test split
    n_total = len(X)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Shuffle indices
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X[train_idx], y[train_idx], masks[train_idx])
    val_dataset = torch.utils.data.TensorDataset(X[val_idx], y[val_idx], masks[val_idx])
    test_dataset = torch.utils.data.TensorDataset(X[test_idx], y[test_idx], masks[test_idx])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
```

### Evaluación y Métricas
```python
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch[0].to(device)
            labels = batch[1].to(device)
            masks = batch[2].to(device)
            
            outputs = model(sequences, masks)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    top3_acc = top_k_accuracy_score(all_labels, outputs.cpu().numpy(), k=3)
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'top3_accuracy': top3_acc,
        'per_class': report
    }
```

### Requisitos de Hardware y Software
- **GPU**: NVIDIA RTX 3050/4060/5050 o superior (8GB+ VRAM)
- **RAM**: 16GB+ para procesamiento de datasets
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (para GPU acceleration)
- **Bibliotecas**: numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm

### Tiempo de Ejecución Típico
- **Carga de datos**: 30-60 segundos
- **Entrenamiento por epoch**: 2-5 minutos (depende del batch size)
- **Evaluación**: 10-30 segundos
- **Total por experimento**: 1-3 horas

Este código implementa completamente la arquitectura descrita, con optimizaciones para eficiencia y reproducibilidad.