import torch
import cv2
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from emotion_classification import EmotionResNet, load_model

classes = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
test_dir = "test"
model = EmotionResNet(num_classes=6)
load_model(model)
device = torch.device("cpu")
model = model.to(device)

    
filepaths, labels = [], []
folds = os.listdir(test_dir)
for fold in folds:
    if fold in classes:
        foldpath = os.path.join(test_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            filepaths.append(str(os.path.join(foldpath, file)))
            labels.append(str(fold))

ts_df = {'filepaths': filepaths, 'labels': labels}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_images_from_filepaths(filepaths):
    inputs = []
    for filepath in filepaths:
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(filepath)
        if image is None:
            print(f"Warning: Could not load image at {filepath}")
            continue
        # Chuyển từ BGR sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Chuyển thành tensor và chuẩn hóa
        inputs.append(image)
    return inputs

inputs = load_images_from_filepaths(ts_df['filepaths'])

ts_df = {'inputs': inputs, 'labels': ts_df['labels']}

predictions = []
true_labels = []
test_correct = 0
total_test = 0


for i in range(0, len(ts_df['inputs'])):
    input = ts_df['inputs'][i]
    input = transform(input).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probabilities)
        preds = classes[predicted_idx]
        
        if preds == labels[i]:
            test_correct += 1
        total_test += 1
        acc = test_correct / total_test
        print(f"Accuracy {i+1}: {acc*100:.2f} %")
        predictions.extend(preds)
        true_labels.extend(labels)

test_accuracy = test_correct / total_test
print(f"Test Accuracy: {test_accuracy*100:.2f} %")