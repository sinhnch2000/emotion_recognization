from fastapi import HTTPException
import os
import urllib.request

# Đường dẫn đến các file mô hình Caffe
CAFFE_PROTO_PATH = "models/deploy.prototxt"
CAFFE_MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# URL để tải file mô hình Caffe nếu chưa có
CAFFE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def download_caffe_models():
    """
    Tải tự động các file mô hình Caffe nếu chưa tồn tại.
    """
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)

    # Tải file deploy.prototxt
    if not os.path.exists(CAFFE_PROTO_PATH):
        print(f"Downloading {CAFFE_PROTO_PATH}...")
        urllib.request.urlretrieve(CAFFE_PROTO_URL, CAFFE_PROTO_PATH)
        print(f"Downloaded {CAFFE_PROTO_PATH}")

    # Tải file res10_300x300_ssd_iter_140000.caffemodel
    if not os.path.exists(CAFFE_MODEL_PATH):
        print(f"Downloading {CAFFE_MODEL_PATH}...")
        urllib.request.urlretrieve(CAFFE_MODEL_URL, CAFFE_MODEL_PATH)
        print(f"Downloaded {CAFFE_MODEL_PATH}")

def validate_image(filename: str) -> bool:
    """Validate if the file is a JPG or PNG image."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(filename.lower())[1]
    return ext in valid_extensions

