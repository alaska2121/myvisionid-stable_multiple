FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p retinaface/
RUN mkdir -p modnet_photographic_portrait_matting/
RUN mkdir -p hivision/creator/weights/

RUN set -x && \
    curl -v -L \
         "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth" \
         -o retinaface/RetinaFace-R50.pth && \
    curl -v -L \
         "https://huggingface.co/yao123/test/resolve/main/modnet_photographic_portrait_matting.ckpt" \
         -o modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt && \
    curl -v -L \
         "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/45df3ded47167a551b8b17b61af4fb6e324051da/birefnet-v1-lite.onnx" \
         -o hivision/creator/weights/birefnet-v1-lite.onnx

COPY . .

RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
