# RunPod Serverless — Whisper large-v3 (faster-whisper + CUDA)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    HF_HOME=/models/huggingface

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /

# Model downloads on first worker start (keeps GitHub Docker build under 30 min limit)
CMD ["python", "-u", "/handler.py"]
