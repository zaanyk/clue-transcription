# Lean CUDA image for RunPod Serverless + faster-whisper large-v3
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    HF_HOME=/models/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r /requirements.txt

COPY handler.py /handler.py
COPY rp_handler.py /rp_handler.py
COPY whisper_engine.py /whisper_engine.py

CMD ["python", "-u", "/handler.py"]
