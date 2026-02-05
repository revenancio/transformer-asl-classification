
import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
from pathlib import Path

# Definir arquitectura (igual que arriba)
class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=96):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.randn(1, max_len, d_model))
        torch.nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderOnlyClassifier(torch.nn.Module):
    def __init__(self, input_dim=128, d_model=256, num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1, num_classes=30, max_seq_len=96):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_dim, d_model)
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len)
        self.dropout = torch.nn.Dropout(dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * (1 - mask_expanded)
            sum_x = x.sum(dim=1)
            lengths = (1 - mask).sum(dim=1).unsqueeze(-1)
            x = sum_x / lengths
        else:
            x = x.mean(dim=1)

        x = self.classifier(x)
        return x

def process_video_to_embeddings(video_path, max_frames=96):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frames_data = []

    while len(frames_data) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(landmarks)
        else:
            frames_data.append([0] * 132)

    cap.release()
    pose.close()

    while len(frames_data) < max_frames:
        frames_data.append([0] * 132)
    frames_data = frames_data[:max_frames]

    embeddings = np.array(frames_data)[:, :128]
    embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min() + 1e-8)
    return embeddings

def predict_video(video_path, model, label_encoder):
    embeddings = process_video_to_embeddings(video_path)
    X_tensor = torch.FloatTensor(embeddings).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
        label = label_encoder.inverse_transform([prediction])[0]
        confidence = torch.softmax(outputs, dim=1).max().item()

    return label, confidence

if __name__ == "__main__":
    # Cargar modelo y label encoder
    model = torch.load('mejor_modelo.pth', map_location='cpu')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Ejemplo de uso
    video_path = 'ejemplo.mp4'  # Cambiar por path real
    if Path(video_path).exists():
        label, conf = predict_video(video_path, model, label_encoder)
        print(f"Predicción: {label}, Confianza: {conf:.4f}")
    else:
        print("Video no encontrado.")
