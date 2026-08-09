# RunPod Serverless — Whisper large-v3 (faster-whisper + CUDA)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    HF_HOME=/models/huggingface \
    CTRANSLATE2_ROOT=/models/ctranslate2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake large-v3 into the image so cold starts don't re-download ~3GB
RUN python - <<'PY'
from faster_whisper import WhisperModel
print("Downloading/caching Whisper large-v3...")
WhisperModel("large-v3", device="cpu", compute_type="int8")
print("Model cached OK")
PY

COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
