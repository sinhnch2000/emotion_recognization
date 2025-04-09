# face_detection.py
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request
from utils import CAFFE_PROTO_PATH, CAFFE_MODEL_PATH

# Tải mô hình YOLOv8 Face Detection từ Hugging Face
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Khởi tạo mô hình YOLO
model = YOLO(model_path)

def detect_faces_caffe(image: np.ndarray) -> list:
    """
    Phát hiện khuôn mặt trong ảnh sử dụng mô hình Caffe SSD.
    
    Args:
        image (np.ndarray): Ảnh đầu vào dưới dạng numpy array (BGR format).
    
    Returns:
        list: Danh sách các bounding box dưới dạng [x, y, w, h].
    """

    # Load mô hình Caffe
    net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_PATH, CAFFE_MODEL_PATH)

    # Chuẩn bị ảnh cho mô hình Caffe
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Đưa ảnh vào mô hình và lấy kết quả
    net.setInput(blob)
    detections = net.forward()

    faces = []
    # Duyệt qua các detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Ngưỡng độ tin cậy
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            faces.append([x, y, w, h])

    return faces

def detect_faces(image: np.ndarray) -> list:
    """
    Phát hiện khuôn mặt trong ảnh sử dụng YOLOv8 Face Detection.
    Nếu YOLOv8 không phát hiện được khuôn mặt, sử dụng mô hình Caffe SSD.
    
    Args:
        image (np.ndarray): Ảnh đầu vào dưới dạng numpy array (BGR format).
    
    Returns:
        list: Danh sách các bounding box dưới dạng [x, y, w, h].
    """
    # Chuyển ảnh từ BGR (OpenCV) sang RGB (YOLO yêu cầu RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuyển numpy array thành PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Dự đoán với YOLOv8
    output = model(pil_image)
    results = Detections.from_ultralytics(output[0])
    
    # Trích xuất bounding boxes từ kết quả YOLOv8
    faces = []
    for detection in results.xyxy:
        x1, y1, x2, y2 = detection[:4]
        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        faces.append([x, y, w, h])
    
    # Nếu YOLOv8 không phát hiện được khuôn mặt, sử dụng Caffe SSD
    if not faces:
        print("YOLOv8 did not detect any faces. Falling back to Caffe SSD...")
        faces = detect_faces_caffe(image)
    
    return faces