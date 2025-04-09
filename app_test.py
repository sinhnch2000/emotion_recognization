import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from PIL import Image
from face_detection import detect_faces
from emotion_classification import EmotionResNet
import json
import base64
from io import BytesIO
from utils import download_caffe_models

download_caffe_models()

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách các class cảm xúc
class_names = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
num_classes = len(class_names)

# Transform cho ảnh đầu vào

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load mô hình đã huấn luyện
model = EmotionResNet(num_classes=num_classes).to(device)
model_path = "models/EmotionCNN_best.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Hàm load ảnh từ input của Gradio
def load_image_from_input(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("No image provided. Please upload an image.")
    # Gradio trả về ảnh dưới dạng numpy array (RGB), chuyển sang BGR cho OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# Hàm dự đoán cảm xúc cho một khuôn mặt
def predict_emotion(face_image: np.ndarray) -> Tuple[np.ndarray, str, float]:
    # Chuyển ảnh từ BGR sang RGB (vì transform sẽ xử lý tiếp)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Áp dụng transform
    face_tensor = transform(face_image).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
    
    return probabilities, predicted_emotion, confidence

# Hàm vẽ ảnh đã crop và bar chart cho từng khuôn mặt
def create_face_chart(face_image: np.ndarray, probabilities: np.ndarray, face_idx: int) -> np.ndarray:
    # Tạo figure với 2 subplot: ảnh khuôn mặt và bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Hiển thị ảnh khuôn mặt đã crop
    ax1.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Face {face_idx + 1}", fontsize=12, color='#2c3e50')
    ax1.axis("off")
    
    # Vẽ bar chart
    ax2.bar(class_names, probabilities, color='#4dabf7')
    ax2.set_title(f"Emotion Probabilities", fontsize=12, color='#2c3e50')
    ax2.set_xlabel("Emotion", fontsize=10, color='#2c3e50')
    ax2.set_ylabel("Probability", fontsize=10, color='#2c3e50')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', labelsize=8, colors='#2c3e50')
    ax2.tick_params(axis='y', labelsize=8, colors='#2c3e50')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    # Lưu figure vào buffer
    plt.tight_layout()
    fig.canvas.draw()
    
    # Chuyển figure thành numpy array
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Lấy dữ liệu RGBA
    result_image = rgba[..., :3]  # Bỏ kênh alpha, chỉ lấy RGB
    
    plt.close(fig)
    
    return result_image

# Hàm chuyển numpy array thành base64 để hiển thị trong HTML
def numpy_to_base64(img: np.ndarray) -> str:
    # Chuyển numpy array thành PIL Image
    img_pil = Image.fromarray(img)
    # Lưu ảnh vào buffer dưới dạng PNG
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    # Chuyển buffer thành base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

# Hàm vẽ bounding box và tạo danh sách ảnh crop + bar chart
def visualize_results(image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[dict]]:
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt
    faces = detect_faces(image)  # Sử dụng ảnh gốc để phát hiện khuôn mặt
    if not faces:
        raise ValueError("No faces detected in the image.")
    
    # Danh sách để lưu thông tin detection
    detection_info_list = []
    # Danh sách để lưu ảnh crop + bar chart
    face_charts = []
    
    # Vẽ bounding box cho từng khuôn mặt
    for i, (x, y, w, h) in enumerate(faces):
        # Cắt khuôn mặt từ ảnh gốc (màu) để dự đoán cảm xúc
        face_image = image[y:y+h, x:x+w]
        if face_image.size == 0:
            continue
        
        # Dự đoán cảm xúc
        probabilities, predicted_emotion, confidence = predict_emotion(face_image)
        
        # Lưu thông tin detection
        detection_info = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "emotion": predicted_emotion,
            "confidence": confidence
        }
        detection_info_list.append(detection_info)
        
        # Vẽ bounding box trên ảnh trắng đen
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Ghi số thứ tự khuôn mặt trên bounding box
        cv2.putText(image_with_boxes, f"Face {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Tạo ảnh crop + bar chart cho khuôn mặt
        face_chart = create_face_chart(face_image, probabilities, i)
        face_charts.append(face_chart)
    
    # Tạo figure cho ảnh gốc với bounding box
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_with_boxes)
    ax.set_title("Detected Faces", fontsize=14, color='#2c3e50')
    ax.axis("off")
    
    # Lưu figure vào buffer
    plt.tight_layout()
    fig.canvas.draw()
    
    # Chuyển figure thành numpy array
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Lấy dữ liệu RGBA
    result_image = rgba[..., :3]  # Bỏ kênh alpha, chỉ lấy RGB
    
    plt.close(fig)
    
    return result_image, face_charts, detection_info_list

# Hàm chính để xử lý ảnh từ Gradio
def process_image(image):
    try:
        # Load ảnh từ input của Gradio
        image = load_image_from_input(image)
        
        # Xử lý và vẽ kết quả
        result_image, face_charts, detection_info_list = visualize_results(image)
        
        # Chuyển detection_info_list thành chuỗi JSON
        detection_info = json.dumps(detection_info_list, indent=2)
        
        # Chuyển các ảnh crop thành base64 để hiển thị trong HTML
        face_charts_base64 = [numpy_to_base64(chart) for chart in face_charts]
        
        # Tạo HTML để hiển thị các ảnh crop
        html_content = """
        <div class="face-container">
            <h3>Cropped Faces with Emotion Probabilities</h3>
        """
        if not face_charts_base64:
            html_content += "<p>No faces detected.</p>"
        else:
            for img_base64 in face_charts_base64:
                html_content += f"""
                <div class="face-image">
                    <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 600px;" />
                </div>
                """
        html_content += "</div>"
        
        return result_image, detection_info, face_charts, html_content
    except Exception as e:
        return None, f"**Error:** {str(e)}", [], "<div class='face-container'><h3>Cropped Faces with Emotion Probabilities</h3><p>No faces detected.</p></div>"

# CSS để tùy chỉnh giao diện
css = """
body {
    background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    font-family: 'Poppins', sans-serif;
    margin: 0;
    padding: 0;
}
h1 {
    color: #1a3c34;
    text-align: center;
    font-size: 2.8em;
    margin-bottom: 10px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}
.description {
    color: #5e6f64;
    text-align: center;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.container {
    max-width: 1300px;
    margin: 0 auto;
    padding: 30px;
}
.input-container, .output-container {
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    padding: 25px;
    margin-bottom: 25px;
    transition: transform 0.3s ease;
}
.input-container:hover, .output-container:hover {
    transform: translateY(-5px);
}
.button-container {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;
}
.submit-btn {
    background: linear-gradient(90deg, #ff6f61 0%, #ff9f43 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 25px !important;
    border-radius: 25px !important;
    font-size: 1.2em !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}
.submit-btn:hover {
    background: linear-gradient(90deg, #ff9f43 0%, #ff6f61 100%) !important;
    transform: scale(1.05) !important;
}
.clear-btn {
    background: linear-gradient(90deg, #a3bffa 0%, #a1c4fd 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 25px !important;
    border-radius: 25px !important;
    font-size: 1.2em !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}
.clear-btn:hover {
    background: linear-gradient(90deg, #a1c4fd 0%, #a3bffa 100%) !important;
    transform: scale(1.05) !important;
}
.output-image {
    border: 3px solid #1a3c34;
    border-radius: 15px;
    transition: border-color 0.3s ease;
}
.output-image:hover {
    border-color: #ff6f61;
}
.face-container {
    max-height: 400px; /* Chiều cao tối đa của container chứa các ảnh */
    overflow-y: auto; /* Bật thanh cuộn dọc */
    padding: 10px;
    border: 2px solid #1a3c34;
    border-radius: 10px;
    margin-top: 20px;
}
.face-image {
    border: 2px solid #1a3e34;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 20px; /* Khoảng cách giữa các ảnh */
}
.detection-info {
    background-color: #f7f9fb;
    border-radius: 10px;
    padding: 15px;
    font-size: 1.1em;
    color: #1a3c34;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    line-height: 1.6;
    font-family: 'Courier New', Courier, monospace;
    white-space: pre-wrap;
}
@media (max-width: 768px) {
    .container {
        padding: 15px;
    }
    .input-container, .output-container {
        padding: 15px;
    }
    h1 {
        font-size: 2em;
    }
    .description {
        font-size: 1em;
    }
    .submit-btn, .clear-btn {
        padding: 10px 20px !important;
        font-size: 1em !important;
    }
    .face-container {
        max-height: 300px; /* Giảm chiều cao trên thiết bị di động */
    }
}
"""

# Tạo giao diện Gradio với gr.Blocks
with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        <h1>Emotion Detection with Bounding Boxes and Probabilities</h1>
        <p class="description">Upload an image to detect faces and predict emotions. The result will show bounding boxes around faces, cropped faces, and their emotion probability bar charts.</p>
        """
    )
    
    # State để lưu trữ danh sách ảnh crop
    face_charts_state = gr.State()
    
    with gr.Row(elem_classes="container"):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-container"):
                input_image = gr.Image(type="numpy", label="Upload an Image", height=300, width=300)
                with gr.Row(elem_classes="button-container"):
                    submit_btn = gr.Button("Submit", elem_classes="submit-btn")
                    clear_btn = gr.Button("Clear", elem_classes="clear-btn")
        
        with gr.Column(scale=2):
            with gr.Group(elem_classes="output-container"):
                output_image = gr.Image(type="numpy", label="Image with Bounding Boxes", height=500, width=800, format="PNG", elem_classes="output-image")
                
                # Sử dụng gr.HTML để hiển thị các ảnh crop động
                face_charts_html = gr.HTML()
                
                detection_info = gr.Textbox(label="Detection Info", elem_classes="detection-info")
    
    # Liên kết các nút với hàm xử lý
    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, detection_info, face_charts_state, face_charts_html]
    )
    clear_btn.click(
        fn=lambda: (None, None, "", [], "<div class='face-container'><h3>Cropped Faces with Emotion Probabilities</h3><p>No faces detected.</p></div>"),
        inputs=None,
        outputs=[input_image, output_image, detection_info, face_charts_state, face_charts_html]
    )

# Chạy giao diện
if __name__ == "__main__":
    demo.launch(share=True)