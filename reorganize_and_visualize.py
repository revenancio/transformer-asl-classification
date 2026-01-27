"""
Script para reorganizar experimentos y generar visualizaciones con etiquetas legibles.
Autor: MLOps Engineer
Fecha: 2026-01-26
"""

import os
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExperimentReorganizer:
    """Clase para reorganizar y visualizar experimentos de ML."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.class_names = self._load_class_names()
        
        # Mapeo de proyectos originales a nuevos nombres
        self.project_mapping = {
            'G4-EMBEDDING FRAME A FRAME GCN': 'embedding_frame_gcn',
            'G4-EMBEDDING FRAME A FRAME UMAP': 'embedding_frame_umap',
            'G4-JSON-NORM': 'json_normalized',
            'G4-QDRANT (Video-Base)': 'qdrant_video_base'
        }
        
    def _load_class_names(self) -> np.ndarray:
        """Carga los nombres de las clases desde el archivo .npy"""
        class_names_path = self.base_path / 'daataset' / 'frame to frame' / 'class_names.npy'
        if class_names_path.exists():
            return np.load(class_names_path, allow_pickle=True)
        else:
            # Si no existe, generar nombres genéricos
            print("⚠️  Archivo de nombres de clases no encontrado. Usando nombres genéricos.")
            return np.array([f"Clase_{i}" for i in range(30)])
    
    def create_directory_structure(self):
        """Crea la estructura de directorios reorganizada."""
        print("\n🔧 Creando estructura de directorios...")
        
        for old_name, new_name in self.project_mapping.items():
            old_path = self.base_path / old_name
            if not old_path.exists():
                continue
                
            new_path = self.base_path / new_name
            
            # Crear subdirectorios para cada experimento
            subdirs = ['results_baseline', 'results_class_weights', 'results_label_smoothing']
            for subdir in subdirs:
                (new_path / subdir).mkdir(parents=True, exist_ok=True)
                
        print("✅ Estructura de directorios creada")
    
    def reorganize_gcn_experiments(self):
        """Reorganiza el proyecto GCN con embeddings frame-a-frame."""
        print("\n📁 Reorganizando proyecto: embedding_frame_gcn")
        
        old_path = self.base_path / 'G4-EMBEDDING FRAME A FRAME GCN'
        new_path = self.base_path / 'embedding_frame_gcn'
        
        if not old_path.exists():
            print("⚠️  Directorio no encontrado")
            return
        
        # Mapeo de archivos del experimento baseline (g8.0)
        baseline_files = {
            'results_baseline': {
                'config.json': 'model_config_g8.0.json',
                'metrics_summary.csv': 'results_g8.0.csv',
                'confusion_matrix.csv': 'confusion_g8.0.csv',
                'confusion_matrix.png': 'confusion_matrix_g8.0.png',
                'per_class_metrics.csv': 'per_class_g8.0.csv',
                'per_class_analysis.png': 'per_class_analysis_g8.0.png',
                'training_curves.png': 'training_curves_g8.0.png',
                'training_log.csv': 'training_log_g8.0.csv',
                'best_model.pt': 'best_model.pt',
                'model_weights.pt': 'model_weights_g8.0.pt'
            }
        }
        
        # Copiar archivos baseline
        self._copy_experiment_files(old_path, new_path / 'results_baseline', baseline_files['results_baseline'])
        
        # Procesar experimentos g8.1 y g8.2 (asumiendo que son class_weights y label_smoothing)
        # Nota: Necesitarás verificar cuál es cuál basándote en los logs
        for subexp in ['g8.1', 'g8.2']:
            subexp_path = old_path / subexp
            if subexp_path.exists():
                # Por ahora, copiaremos g8.1 a class_weights y g8.2 a label_smoothing
                target = 'results_class_weights' if subexp == 'g8.1' else 'results_label_smoothing'
                self._copy_subexperiment_files(subexp_path, new_path / target)
    
    def reorganize_umap_experiments(self):
        """Reorganiza el proyecto UMAP con embeddings frame-a-frame."""
        print("\n📁 Reorganizando proyecto: embedding_frame_umap")
        
        old_path = self.base_path / 'G4-EMBEDDING FRAME A FRAME UMAP'
        new_path = self.base_path / 'embedding_frame_umap'
        
        if not old_path.exists():
            print("⚠️  Directorio no encontrado")
            return
        
        # Baseline (principal)
        baseline_files = {
            'config.json': 'model_config_umap.json',
            'metrics_summary.csv': 'results_umap.csv',
            'confusion_matrix.csv': 'confusion_umap.csv',
            'confusion_matrix.png': 'confusion_matrix_umap.png',
            'per_class_metrics.csv': 'per_class_umap.csv',
            'per_class_analysis.png': 'per_class_analysis_umap.png',
            'training_curves.png': 'training_curves_umap.png',
            'training_log.csv': 'training_log_umap.csv'
        }
        
        self._copy_experiment_files(old_path, new_path / 'results_baseline', baseline_files)
        
        # Class weights (g6)
        g6_path = old_path / 'g6_class_weights'
        if g6_path.exists():
            self._copy_subexperiment_files(g6_path, new_path / 'results_class_weights')
        
        # Label smoothing (g7)
        g7_path = old_path / 'g7_label_smooth'
        if g7_path.exists():
            self._copy_subexperiment_files(g7_path, new_path / 'results_label_smoothing')
    
    def reorganize_json_experiments(self):
        """Reorganiza el proyecto JSON normalizado."""
        print("\n📁 Reorganizando proyecto: json_normalized")
        
        old_path = self.base_path / 'G4-JSON-NORM'
        new_path = self.base_path / 'json_normalized'
        
        if not old_path.exists():
            print("⚠️  Directorio no encontrado")
            return
        
        baseline_files = {
            'config.json': 'model_config_g5.0.json',
            'metrics_summary.csv': 'results_g5.0.csv',
            'confusion_matrix.csv': 'confusion_g5.0.csv',
            'confusion_matrix.png': 'confusion_matrix_g5.png',
            'per_class_metrics.csv': 'per_class_g5.0.csv',
            'per_class_analysis.png': 'per_class_analysis_g5.png',
            'training_curves.png': 'training_curves_g5.png',
            'training_log.csv': 'training_log_g5.0.csv',
            'best_model.pt': 'best_model.pt',
            'model_weights.pt': 'model_weights.pt'
        }
        
        self._copy_experiment_files(old_path, new_path / 'results_baseline', baseline_files)
        
        for subexp in ['g5.1', 'g5.2']:
            subexp_path = old_path / subexp
            if subexp_path.exists():
                target = 'results_class_weights' if subexp == 'g5.1' else 'results_label_smoothing'
                self._copy_subexperiment_files(subexp_path, new_path / target)
    
    def _copy_experiment_files(self, source_dir: Path, target_dir: Path, file_mapping: Dict[str, str]):
        """Copia archivos de experimento con nuevo nombre."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for new_name, old_name in file_mapping.items():
            source_file = source_dir / old_name
            target_file = target_dir / new_name
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                print(f"  ✓ Copiado: {new_name}")
            else:
                print(f"  ⚠️  No encontrado: {old_name}")
    
    def _copy_subexperiment_files(self, source_dir: Path, target_dir: Path):
        """Copia archivos de subexperimentos."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_mapping = {
            'confusion_matrix.csv': 'confusion.csv',
            'metrics_summary.csv': 'results.csv',
            'training_log.csv': 'training_log.csv'
        }
        
        for new_name, old_name in file_mapping.items():
            source_file = source_dir / old_name
            target_file = target_dir / new_name
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
    
    def generate_visualizations_for_experiment(self, project_name: str, experiment_type: str):
        """Genera visualizaciones completas para un experimento."""
        print(f"\n🎨 Generando visualizaciones: {project_name}/{experiment_type}")
        
        exp_path = self.base_path / project_name / experiment_type
        
        if not exp_path.exists():
            print(f"  ⚠️  Directorio no existe: {exp_path}")
            return
        
        # 1. Generar/actualizar metrics_summary.csv
        self._ensure_metrics_summary(exp_path)
        
        # 2. Generar matriz de confusión con etiquetas
        self._generate_confusion_matrix_plot(exp_path)
        
        # 3. Generar curvas de aprendizaje
        self._generate_learning_curves(exp_path)
        
        # 4. Generar análisis por clase
        self._generate_per_class_analysis(exp_path)
        
        print(f"  ✅ Visualizaciones completadas")
    
    def _ensure_metrics_summary(self, exp_path: Path):
        """Asegura que metrics_summary.csv exista con el formato correcto."""
        metrics_file = exp_path / 'metrics_summary.csv'
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            
            # Verificar formato
            if 'Metric' in df.columns and 'Value' in df.columns:
                print("  ✓ metrics_summary.csv ya tiene el formato correcto")
                return
        
        # Si no existe o tiene formato incorrecto, buscar en config.json
        config_file = exp_path / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            metrics_data = {
                'Metric': ['Accuracy', 'Macro-F1', 'Top-3 Accuracy', 'Test Loss'],
                'Value': [
                    config.get('test_accuracy', 0.0),
                    config.get('test_macro_f1', 0.0),
                    config.get('test_top3_accuracy', 0.0),
                    config.get('test_loss', 0.0)
                ]
            }
            
            df = pd.DataFrame(metrics_data)
            df.to_csv(metrics_file, index=False)
            print(f"  ✓ Creado metrics_summary.csv")
    
    def _generate_confusion_matrix_plot(self, exp_path: Path):
        """Genera gráfico de matriz de confusión con etiquetas de texto."""
        confusion_csv = exp_path / 'confusion_matrix.csv'
        
        if not confusion_csv.exists():
            print(f"  ⚠️  No se encontró confusion_matrix.csv")
            return
        
        # Leer matriz de confusión
        cm = pd.read_csv(confusion_csv, header=None).values
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(20, 18))
        
        # Usar solo las primeras N clases según el tamaño de la matriz
        num_classes = cm.shape[0]
        class_labels = self.class_names[:num_classes]
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, 
                    yticklabels=class_labels,
                    cbar_kws={'label': 'Número de muestras'},
                    ax=ax)
        
        ax.set_xlabel('Predicción', fontsize=14, fontweight='bold')
        ax.set_ylabel('Etiqueta Real', fontsize=14, fontweight='bold')
        ax.set_title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
        
        # Rotar etiquetas para mejor legibilidad
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        output_file = exp_path / 'confusion_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Matriz de confusión generada")
    
    def _generate_learning_curves(self, exp_path: Path):
        """Genera curvas de aprendizaje (Loss y Accuracy)."""
        training_log = exp_path / 'training_log.csv'
        
        if not training_log.exists():
            print(f"  ⚠️  No se encontró training_log.csv")
            return
        
        df = pd.read_csv(training_log)
        
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Loss
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
            axes[0].plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Época', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
            axes[0].set_title('Curva de Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Accuracy
        if 'train_acc' in df.columns and 'val_acc' in df.columns:
            axes[1].plot(df['epoch'], df['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
            axes[1].plot(df['epoch'], df['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Época', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_title('Curva de Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = exp_path / 'training_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Curvas de aprendizaje generadas")
    
    def _generate_per_class_analysis(self, exp_path: Path):
        """Genera análisis detallado por clase con etiquetas."""
        per_class_csv = exp_path / 'per_class_metrics.csv'
        
        if not per_class_csv.exists():
            print(f"  ⚠️  No se encontró per_class_metrics.csv")
            return
        
        df = pd.read_csv(per_class_csv)
        
        # Si la primera columna es índice numérico, reemplazar con nombres
        if df.columns[0] == 'Unnamed: 0' or df.iloc[:, 0].dtype in [np.int64, np.int32]:
            class_indices = df.iloc[:, 0].values
            # Filtrar filas que no sean métricas agregadas
            mask = ~df.iloc[:, 0].isin(['accuracy', 'macro avg', 'weighted avg'])
            df_filtered = df[mask].copy()
            
            # Reemplazar índices con nombres
            class_indices_filtered = df_filtered.iloc[:, 0].astype(int).values
            df_filtered.iloc[:, 0] = self.class_names[class_indices_filtered]
            df_filtered.rename(columns={df_filtered.columns[0]: 'Clase'}, inplace=True)
        else:
            df_filtered = df[~df.iloc[:, 0].isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        
        # Asegurar que tenemos las columnas necesarias
        metrics = ['precision', 'recall', 'f1-score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in df_filtered.columns:
                ax = axes[idx]
                
                # Crear gráfico de barras
                y_pos = np.arange(len(df_filtered))
                values = df_filtered[metric].values
                
                bars = ax.barh(y_pos, values, color=color, alpha=0.7, edgecolor='black')
                
                # Configurar etiquetas y título
                ax.set_yticks(y_pos)
                ax.set_yticklabels(df_filtered.iloc[:, 0].values, fontsize=9)
                ax.set_xlabel(metric.capitalize(), fontsize=12, fontweight='bold')
                ax.set_title(f'{metric.capitalize()} por Clase', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1.0)
                ax.grid(True, axis='x', alpha=0.3)
                
                # Agregar valores en las barras
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(value + 0.02, i, f'{value:.3f}', 
                           va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = exp_path / 'per_class_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar versión actualizada del CSV con nombres
        output_csv = exp_path / 'per_class_metrics.csv'
        df_filtered.to_csv(output_csv, index=False)
        
        print(f"  ✓ Análisis por clase generado")
    
    def generate_comparison_report(self, project_name: str):
        """Genera reporte comparativo entre los tres experimentos."""
        print(f"\n📊 Generando reporte comparativo: {project_name}")
        
        project_path = self.base_path / project_name
        
        if not project_path.exists():
            print(f"  ⚠️  Proyecto no encontrado")
            return
        
        experiments = ['results_baseline', 'results_class_weights', 'results_label_smoothing']
        comparison_data = []
        
        for exp in experiments:
            exp_path = project_path / exp
            metrics_file = exp_path / 'metrics_summary.csv'
            
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                metrics_dict = dict(zip(df['Metric'], df['Value']))
                metrics_dict['Experiment'] = exp.replace('results_', '').replace('_', ' ').title()
                comparison_data.append(metrics_dict)
        
        if not comparison_data:
            print("  ⚠️  No se encontraron datos de métricas")
            return
        
        # Crear DataFrame comparativo
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df[['Experiment', 'Accuracy', 'Macro-F1', 'Top-3 Accuracy', 'Test Loss']]
        
        # Guardar CSV
        output_csv = project_path / 'experiments_comparison.csv'
        comparison_df.to_csv(output_csv, index=False)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'Macro-F1', 'Top-3 Accuracy', 'Test Loss']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = comparison_df[metric].values
            x_pos = np.arange(len(comparison_df))
            
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(comparison_df['Experiment'], rotation=15, ha='right')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'Comparación: {metric}', fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Agregar valores en las barras
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Comparación de Experimentos - {project_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_png = project_path / 'experiments_comparison.png'
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Reporte comparativo generado")
        print(f"\n{comparison_df.to_string(index=False)}\n")
    
    def run_full_reorganization(self):
        """Ejecuta el proceso completo de reorganización."""
        print("="*80)
        print("🚀 INICIANDO REORGANIZACIÓN Y GENERACIÓN DE REPORTES")
        print("="*80)
        
        # 1. Crear estructura
        self.create_directory_structure()
        
        # 2. Reorganizar cada proyecto
        self.reorganize_gcn_experiments()
        self.reorganize_umap_experiments()
        self.reorganize_json_experiments()
        
        # 3. Generar visualizaciones para cada experimento
        projects = ['embedding_frame_gcn', 'embedding_frame_umap', 'json_normalized']
        experiments = ['results_baseline', 'results_class_weights', 'results_label_smoothing']
        
        for project in projects:
            for experiment in experiments:
                self.generate_visualizations_for_experiment(project, experiment)
        
        # 4. Generar reportes comparativos
        for project in projects:
            self.generate_comparison_report(project)
        
        print("\n" + "="*80)
        print("✅ REORGANIZACIÓN COMPLETADA EXITOSAMENTE")
        print("="*80)
        
        # Resumen final
        print("\n📋 RESUMEN DE LA ESTRUCTURA:")
        for project in projects:
            print(f"\n{project}/")
            for exp in experiments:
                exp_path = self.base_path / project / exp
                if exp_path.exists():
                    files = list(exp_path.glob('*'))
                    print(f"  ├── {exp}/ ({len(files)} archivos)")


def main():
    """Función principal."""
    base_path = r"c:\Users\Los milluelitos repo\Desktop\experimento tesis\transformer-asl-classification"
    
    reorganizer = ExperimentReorganizer(base_path)
    reorganizer.run_full_reorganization()


if __name__ == "__main__":
    main()
