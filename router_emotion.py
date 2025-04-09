from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import cv2
import numpy as np
from face_detection import detect_faces
from emotion_classification import classify_emotion
from utils import validate_image, download_caffe_models

download_caffe_models()

router = APIRouter()

@router.post("/emotion", response_model=List[dict])
async def analyze_emotion(file: UploadFile = File(...)):
    """
    Analyze emotions from an uploaded image.
    Returns a list of detected faces with bounding box, emotion, and confidence.
    """
    # Validate image format
    if not validate_image(file.filename):
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPG and PNG are supported.")

    # Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    # Check resolution
    height, width = image.shape[:2]
    if width > 1920 or height > 1080:
        raise HTTPException(status_code=400, detail="Image resolution exceeds 1920x1080.")

    # Detect faces
    faces = detect_faces(image)
    if not faces:
        return []

    # Analyze emotions
    results = []
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        emotion, confidence = classify_emotion(face_roi)
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": float(confidence)
        })

    return results