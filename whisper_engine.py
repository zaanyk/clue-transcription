import base64
import os
import tempfile
import time

import requests
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))

_model = None


def get_model():
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


def _load_audio_bytes(job_input):
    if job_input.get("audio_base64"):
        return base64.b64decode(job_input["audio_base64"])

    audio_url = job_input.get("audio_url")
    if audio_url:
        resp = requests.get(audio_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    raise ValueError("Provide audio_base64 or audio_url")


def transcribe_job(job_input):
    started = time.time()
    language = job_input.get("language") or "uk"
    task = job_input.get("task") or "transcribe"
    beam_size = int(job_input.get("beam_size") or 5)
    vad_filter = bool(job_input.get("vad_filter", True))

    audio_path = None
    try:
        audio_bytes = _load_audio_bytes(job_input)
        if not audio_bytes:
            raise ValueError("Empty audio payload")

        fd, audio_path = tempfile.mkstemp(suffix=".audio")
        os.close(fd)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        whisper = get_model()
        segments, info = whisper.transcribe(
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
