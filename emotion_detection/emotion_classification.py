import torch
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
import torch.nn.functional as F

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
# Khởi tạo mô hình
model = EmotionResNet(num_classes=6)
device = torch.device("cpu")
model = model.to(device)

# Transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Danh sách cảm xúc (labels)
class_names = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

def load_model(model, model_path="models/EmotionCNN_best.pth"):
    """Load pre-trained model weights."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {model_path}")
        model.is_loaded = True
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

def classify_emotion(face_roi: np.ndarray) -> tuple:
    """Classify emotion from a face region of interest (ROI)."""
    if not hasattr(model, 'is_loaded'):
        load_model(model)
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face = transform(face_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probabilities)
        emotion = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
    return emotion, confidence