# # # Step 1: Build the image
# # docker build -t emotion:v1 .

# # # # Step 2: Run the container
# # docker run -d -p 8000:8000 --restart=always --name emotion_container emotion:v1

# docker tag 685329e51159 lightning7777777777/emotion_v1:v1
# docker push lightning7777777777/emotion_v1:v1

 

FROM python:3.12-slim

# Cài đặt các dependency hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libjpeg62-turbo-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Thêm --no-cache-dir để giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
