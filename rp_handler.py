"""
RunPod Serverless handler — Whisper large-v3 (faster-whisper).

Input (job["input"]):
  {
    "audio_base64": "<base64>",   # required (or audio_url)
    "audio_url": "https://...",   # alternative to base64
    "language": "uk",             # optional, default uk
    "task": "transcribe"          # optional: transcribe | translate
  }

Output:
  { "text": "...", "language": "uk", "model": "large-v3", "duration_sec": 12.3 }
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from typing import Any

import requests
import runpod
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"Loading Whisper model={MODEL_NAME} device={DEVICE} compute={COMPUTE_TYPE}")
        _model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
        )
        print("Whisper model ready")
    return _model


def _write_audio_temp(audio_bytes: bytes, suffix: str = ".audio") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def _load_audio_bytes(job_input: dict[str, Any]) -> bytes:
    if job_input.get("audio_base64"):
        return base64.b64decode(job_input["audio_base64"])

    audio_url = job_input.get("audio_url")
    if audio_url:
        resp = requests.get(audio_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    raise ValueError("Provide audio_base64 or audio_url")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    job_input = job.get("input") or {}

    language = job_input.get("language") or "uk"
    task = job_input.get("task") or "transcribe"
    beam_size = int(job_input.get("beam_size") or 5)
    vad_filter = bool(job_input.get("vad_filter", True))

    audio_path = None
    try:
        audio_bytes = _load_audio_bytes(job_input)
        if not audio_bytes:
            raise ValueError("Empty audio payload")

        audio_path = _write_audio_temp(audio_bytes)
        model = get_model()

        segments, info = model.transcribe(
            audio_path,
            language=None if language in ("auto", "") else language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            raise RuntimeError("Empty transcript from Whisper")

        return {
            "text": text,
            "language": getattr(info, "language", language),
            "model": MODEL_NAME,
            "duration_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


# Warm model on worker start (reduces first-request latency after cold start)
try:
    get_model()
except Exception as e:
    print(f"Model preload failed (will retry on first job): {e}")

runpod.serverless.start({"handler": handler})
