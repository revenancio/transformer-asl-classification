import numpy as np
import pandas as pd
from copy import deepcopy
from src.config import *

def interpolar_manos_faltantes(frames_data):
    """
    Rellena los frames donde las manos aparecen como listas vacías [] 
    usando interpolación lineal basada en los frames vecinos.
    """
    # Convertimos la lista de diccionarios a un formato más amigable para Pandas
    # Extraemos las series temporales de cada punto de la mano
    
    parts = ['left_hand_landmarks', 'right_hand_landmarks']
    # Asumimos 21 puntos por mano (estándar mediapipe)
    num_points = 21 
    
    processed_frames = deepcopy(frames_data)
    
    # Creamos un DataFrame temporal para cada coordenada de cada punto de cada mano
    for part in parts:
        # 1. Construir matriz temporal (Frames x Puntos x 3)
        # Inicializamos con NaN para poder interpolar después
        n_frames = len(processed_frames)
        data_matrix = np.full((n_frames, num_points, 3), np.nan)
        
        for t, frame in enumerate(processed_frames):
            landmarks = frame.get('landmarks', {}).get(part, [])
            if landmarks: # Si la lista no está vacía
                for i, point in enumerate(landmarks):
                    data_matrix[t, i, 0] = point.get('x', np.nan)
                    data_matrix[t, i, 1] = point.get('y', np.nan)
                    data_matrix[t, i, 2] = point.get('z', np.nan)
        
        # 2. Interpolar usando Pandas
        # Convertimos a 2D para interpolar (flatten) y luego regresamos a 3D
        # Estructura: Filas=Frames, Columnas=(Punto0_x, Punto0_y... Punto20_z)
        flat_data = data_matrix.reshape(n_frames, -1) 
        df = pd.DataFrame(flat_data)
        
        # Interpolación lineal, limit_direction='both' rellena los bordes también
        df_interpolated = df.interpolate(method='linear', limit_direction='both', axis=0)
        
        # Si todo el video no tiene manos, fillna(0) para evitar errores, 
        # aunque esos videos deberían haberse filtrado antes.
        df_interpolated = df_interpolated.fillna(0)
        
        reconstructed_data = df_interpolated.values.reshape(n_frames, num_points, 3)
        
        # 3. Guardar de vuelta en la estructura de lista
        for t in range(n_frames):
            new_landmarks = []
            for i in range(num_points):
                new_landmarks.append({
                    'x': reconstructed_data[t, i, 0],
                    'y': reconstructed_data[t, i, 1],
                    'z': reconstructed_data[t, i, 2]
                })
            processed_frames[t]['landmarks'][part] = new_landmarks
            
    return processed_frames

def normalizacion_espacial(frames_data):
    """
    1. Traslada todos los puntos para que el centro de los hombros sea (0,0,0).
    2. Escala todos los puntos para que el ancho de hombros sea 1.0.
    """
    normalized_frames = deepcopy(frames_data)
    
    for frame in normalized_frames:
        lms = frame.get('landmarks', {})
        pose = lms.get('pose_landmarks', [])
        
        # Necesitamos la pose para normalizar. Si no hay pose, saltamos (o usamos la anterior).
        if not pose or len(pose) < 33:
            continue
            
        # --- PASO 1: TRASLADO (CENTRADO) ---
        # Obtener coordenadas de hombros
        left_shoulder = pose[MEDIAPIPE_SHOULDER_LEFT]
        right_shoulder = pose[MEDIAPIPE_SHOULDER_RIGHT]
        
        # Calcular centrooide (punto medio entre hombros)
        centroid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        centroid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        centroid_z = (left_shoulder['z'] + right_shoulder['z']) / 2
        
        # --- PASO 2: ESCALADO ---
        # Calcular ancho de hombros (distancia Euclideana)
        shoulder_width = np.sqrt(
            (left_shoulder['x'] - right_shoulder['x'])**2 +
            (left_shoulder['y'] - right_shoulder['y'])**2 +
            (left_shoulder['z'] - right_shoulder['z'])**2
        )
        
        # Factor de escala: queremos que el ancho sea 1.0 (o cualquier constante)
        # Evitamos división por cero
        scale_factor = 1.0 / shoulder_width if shoulder_width > 0 else 1.0
        
        # Aplicar transformación a TODOS los landmarks (pose, cara, manos)
        all_keys = ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']
        
        for key in all_keys:
            points = lms.get(key, [])
            for point in points:
                # Trasladar
                point['x'] -= centroid_x
                point['y'] -= centroid_y
                point['z'] -= centroid_z
                
                # Escalar
                point['x'] *= scale_factor
                point['y'] *= scale_factor
                point['z'] *= scale_factor
                
    return normalized_frames

def normalizacion_temporal_resample(frames_data, target_frames=TARGET_FRAMES):
    """
    Ajusta la secuencia a `target_frames` usando interpolación lineal.
    """
    original_frames = len(frames_data)
    if original_frames == 0:
        return []
    
    if original_frames == target_frames:
        return frames_data
    
    # Índices originales y objetivo
    original_indices = np.linspace(0, original_frames - 1, num=original_frames)
    target_indices = np.linspace(0, original_frames - 1, num=target_frames)
    
    resampled_data = []
    
    # Lista de claves a procesar
    keys_to_process = ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']
    
    # Pre-cálculo para eficiencia: Extraer todo a numpy
    # (Simplificado: iteramos estructura para mantener legibilidad)
    
    # Estructura del nuevo frame
    for t_idx in range(target_frames):
        new_frame = {'frame_index': t_idx, 'landmarks': {}}
        t_val = target_indices[t_idx] # El tiempo "virtual" actual
        
        # Encontrar frames vecinos (indices enteros) para interpolar
        # np.searchsorted busca dónde encajaría t_val
        idx_right = np.searchsorted(original_indices, t_val, side='left')
        idx_left = max(0, idx_right - 1)
        idx_right = min(original_frames - 1, idx_right)
        
        # Factor de peso para la interpolación (alpha)
        if idx_left == idx_right:
            alpha = 0
        else:
            time_left = original_indices[idx_left]
            time_right = original_indices[idx_right]
            alpha = (t_val - time_left) / (time_right - time_left)
            
        frame_left = frames_data[idx_left]['landmarks']
        frame_right = frames_data[idx_right]['landmarks']
        
        for key in keys_to_process:
            points_left = frame_left.get(key, [])
            points_right = frame_right.get(key, [])
            
            # Asumimos que tras el paso 1 (interpolación de manos), las listas tienen tamaño correcto
            # Si una está vacía (ej. cara no detectada en todo el video), se queda vacía.
            if not points_left or not points_right:
                new_frame['landmarks'][key] = []
                continue
                
            new_points = []
            for i in range(len(points_left)):
                p_l = points_left[i]
                p_r = points_right[i]
                
                new_p = {
                    'x': p_l['x'] + (p_r['x'] - p_l['x']) * alpha,
                    'y': p_l['y'] + (p_r['y'] - p_l['y']) * alpha,
                    'z': p_l['z'] + (p_r['z'] - p_l['z']) * alpha
                }
                new_points.append(new_p)
            new_frame['landmarks'][key] = new_points
            
        resampled_data.append(new_frame)
        
    return resampled_data

def frame_to_vector(landmarks):
    """
    Versión Inteligente: Detecta si los datos vienen filtrados (índices 0-N)
    o si son raw mediapipe (índices originales).
    """
    vector = []
    
    parts_config = [
        ('pose_landmarks', INDICES_POSE),
        ('face_landmarks', INDICES_FACE),
        ('left_hand_landmarks', INDICES_HAND),
        ('right_hand_landmarks', INDICES_HAND)
    ]
    
    for key, indices_to_extract in parts_config:
        points_list = landmarks.get(key, [])
        
        # 1. Si no hay datos, rellenamos con ceros
        if not points_list:
            vector.extend([0.0] * (len(indices_to_extract) * 3))
            continue

        # 2. DETECCIÓN AUTOMÁTICA DE FORMATO
        # Si la lista tiene el mismo tamaño que los puntos pedidos (ej. Pose tiene 6 y pedimos 6)
        # significa que el JSON YA VIENE RECORTADO. Ignoramos los índices y tomamos todo en orden.
        if len(points_list) == len(indices_to_extract):
            for p in points_list:
                vector.extend([
                    p.get('x', 0.0),
                    p.get('y', 0.0),
                    p.get('z', 0.0)
                ])
        
        # Caso especial Pose: A veces trae 6 puntos exactos aunque pidamos índices altos
        elif key == 'pose_landmarks' and len(points_list) == 6:
             for p in points_list:
                vector.extend([
                    p.get('x', 0.0),
                    p.get('y', 0.0),
                    p.get('z', 0.0)
                ])
                
        # 3. Formato Raw (MediaPipe Original)
        # Aquí sí usamos los índices específicos (11, 402, etc.)
        else:
            for idx in indices_to_extract:
                if idx < len(points_list):
                    p = points_list[idx]
                    vector.extend([
                        p.get('x', 0.0),
                        p.get('y', 0.0),
                        p.get('z', 0.0)
                    ])
                else:
                    # Este era el error: antes caía siempre aquí
                    vector.extend([0.0, 0.0, 0.0])
            
    return vector

def normalizacion_temporal_hibrida(frames_data, target_frames):
    """
    Estrategia Híbrida para LSTMs/Transformers:
    - Si frames > target: Subsampling (salta frames).
    - Si frames < target: Padding (rellena con vacío).
    
    Retorna:
        - data_out: Lista de frames ajustada al target.
        - mask: Array (target_frames,) con 1.0 (Real) y 0.0 (Padding).
    """
    original_frames = len(frames_data)
    
    # Caso 0: Video vacío
    if original_frames == 0:
        return [], np.zeros(target_frames).tolist()
    
    # --- CASO A: SUBSAMPLING (Video Largo) ---
    if original_frames > target_frames:
        indices = np.linspace(0, original_frames - 1, num=target_frames).astype(int)
        resampled_data = [deepcopy(frames_data[i]) for i in indices]
        # Todo es real, máscara de unos
        mask = np.ones(target_frames, dtype=np.float32).tolist()
        return resampled_data, mask

    # --- CASO B: PADDING (Video Corto) ---
    else:
        data_padded = deepcopy(frames_data)
        padding_size = target_frames - original_frames
        
        # Rellenar con estructuras vacías
        for i in range(padding_size):
            empty_frame = {
                'frame_index': original_frames + i,
                'landmarks': {
                    'pose_landmarks': [], 'face_landmarks': [],
                    'left_hand_landmarks': [], 'right_hand_landmarks': []
                }
            }
            data_padded.append(empty_frame)
            
        # Máscara: 1.0 para originales, 0.0 para relleno
        mask = np.zeros(target_frames, dtype=np.float32)
        mask[:original_frames] = 1.0
        
        return data_padded, mask.tolist()