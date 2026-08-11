import base64
import json
import os
import subprocess
import tempfile
import time

import requests
from faster_whisper import WhisperModel

DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
DEFAULT_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

# Telephony convention: left/ch0 = agent (manager), right/ch1 = customer (client)
DEFAULT_MANAGER_CHANNEL = int(os.getenv("MANAGER_AUDIO_CHANNEL", "0"))
DEFAULT_CLIENT_CHANNEL = int(os.getenv("CLIENT_AUDIO_CHANNEL", "1"))

# Faster VAD: skip more silence on long telephony calls
DEFAULT_VAD_PARAMETERS = {
    "min_silence_duration_ms": int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "500")),
    "speech_pad_ms": int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "200")),
}

_models = {}


def get_model(model_name: str = None, compute_type: str = None):
    name = model_name or DEFAULT_MODEL_NAME
    ctype = compute_type or DEFAULT_COMPUTE_TYPE
    key = f"{name}|{DEVICE}|{ctype}"
    if key not in _models:
        print(f"Loading Whisper model={name} device={DEVICE} compute={ctype}")
        _models[key] = WhisperModel(
            name,
            device=DEVICE,
            compute_type=ctype,
            cpu_threads=CPU_THREADS,
        )
        print(f"Whisper model ready: {name}")
    return _models[key], name


def _load_audio_bytes(job_input):
    if job_input.get("audio_base64"):
        return base64.b64decode(job_input["audio_base64"])

    audio_url = job_input.get("audio_url")
    if audio_url:
        resp = requests.get(audio_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    raise ValueError("Provide audio_base64 or audio_url")


def _probe_channels(audio_path: str) -> int:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "json",
                audio_path,
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        data = json.loads(out)
        channels = int(data["streams"][0].get("channels") or 1)
        return max(1, channels)
    except Exception as exc:
        print(f"ffprobe channels failed, assuming mono: {exc}")
        return 1


def _extract_mono_channel(src_path: str, channel_index: int, dest_path: str) -> None:
    # pan=mono|c0=cN keeps only channel N as mono PCM WAV
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-af",
            f"pan=mono|c0=c{channel_index}",
            "-ac",
            "1",
            dest_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _transcribe_file(
    audio_path: str,
    language: str,
    task: str,
    beam_size: int,
    vad_filter: bool,
    model_name: str = None,
    compute_type: str = None,
):
    whisper, resolved_model = get_model(model_name, compute_type)
    segments_iter, info = whisper.transcribe(
        audio_path,
        language=None if language in ("auto", "") else language,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=DEFAULT_VAD_PARAMETERS if vad_filter else None,
        condition_on_previous_text=False,
    )

    segments = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "start": float(seg.start or 0),
                "end": float(seg.end or 0),
            }
        )
    return segments, info, resolved_model


def _format_labeled_text(segments: list) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or ""
        label = (
            "Менеджер"
            if speaker == "manager"
            else "Клієнт"
            if speaker == "client"
            else speaker
        )
        body = (seg.get("text") or "").strip()
        if not body:
            continue
        lines.append(f"{label}: {body}" if label else body)
    return "\n\n".join(lines)


def _as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def transcribe_job(job_input):
    started = time.time()
    language = job_input.get("language") or "uk"
    task = job_input.get("task") or "transcribe"
    beam_size = int(job_input.get("beam_size") or DEFAULT_BEAM_SIZE)
    vad_filter = _as_bool(job_input.get("vad_filter"), True)
    model_name = job_input.get("model") or DEFAULT_MODEL_NAME
    compute_type = job_input.get("compute_type") or DEFAULT_COMPUTE_TYPE
    manager_channel = int(job_input.get("manager_channel", DEFAULT_MANAGER_CHANNEL))
    client_channel = int(job_input.get("client_channel", DEFAULT_CLIENT_CHANNEL))

    audio_path = None
    channel_paths = []
    resolved_model = model_name
    try:
        audio_bytes = _load_audio_bytes(job_input)
        if not audio_bytes:
            raise ValueError("Empty audio payload")

        fd, audio_path = tempfile.mkstemp(suffix=".audio")
        os.close(fd)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        channels = _probe_channels(audio_path)
        print(
            f"Audio channels={channels} model={model_name} beam_size={beam_size} "
            f"vad_filter={vad_filter} compute={compute_type}"
        )

        if channels >= 2:
            # Stereo (or more): each channel → separate Whisper pass on RunPod
            role_by_channel = {
                manager_channel: "manager",
                client_channel: "client",
            }
            # Fallback if misconfigured: ch0 manager, ch1 client
            if manager_channel == client_channel:
                role_by_channel = {0: "manager", 1: "client"}

            labeled = []
            info = None
            for ch in sorted(role_by_channel.keys()):
                if ch < 0 or ch >= channels:
                    continue
                fd_ch, ch_path = tempfile.mkstemp(suffix=f".ch{ch}.wav")
                os.close(fd_ch)
                channel_paths.append(ch_path)
                _extract_mono_channel(audio_path, ch, ch_path)
                segs, info, resolved_model = _transcribe_file(
                    ch_path,
                    language,
                    task,
                    beam_size,
                    vad_filter,
                    model_name=model_name,
                    compute_type=compute_type,
                )
                role = role_by_channel[ch]
                for s in segs:
                    labeled.append(
                        {
                            "speaker": role,
                            "text": s["text"],
                            "start": s["start"],
                            "end": s["end"],
                            "startMs": int(round(s["start"] * 1000)),
                            "endMs": int(round(s["end"] * 1000)),
                        }
                    )

            labeled.sort(key=lambda s: (s["start"], 0 if s["speaker"] == "manager" else 1))
            text = _format_labeled_text(labeled)
            if not text:
                raise RuntimeError("Empty transcript from Whisper (stereo)")

            return {
                "text": text,
                "segments": labeled,
                "channels": channels,
                "speaker_mode": "stereo",
                "language": getattr(info, "language", language) if info else language,
                "model": resolved_model,
                "beam_size": beam_size,
                "duration_sec": round(time.time() - started, 3),
            }

        # Mono: single Whisper pass (no channel-based speaker split)
        segs, info, resolved_model = _transcribe_file(
            audio_path,
            language,
            task,
            beam_size,
            vad_filter,
            model_name=model_name,
            compute_type=compute_type,
        )
        text = " ".join(s["text"] for s in segs).strip()
        if not text:
            raise RuntimeError("Empty transcript from Whisper")

        segments = [
            {
                "speaker": "unknown",
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                "startMs": int(round(s["start"] * 1000)),
                "endMs": int(round(s["end"] * 1000)),
            }
            for s in segs
        ]

        return {
            "text": text,
            "segments": segments,
            "channels": 1,
            "speaker_mode": "mono",
            "language": getattr(info, "language", language),
            "model": resolved_model,
            "beam_size": beam_size,
            "duration_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        for path in [audio_path, *channel_paths]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
