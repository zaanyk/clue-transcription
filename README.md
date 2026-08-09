# clue-transcription

RunPod Serverless worker: **Whisper large-v3** via faster-whisper (CUDA).

## Quick start

1. Create a **private** GitHub repo named `clue-transcription`.
2. Push this folder:
   ```bash
   cd clue-transcription
   git init
   git add .
   git commit -m "Initial Whisper large-v3 RunPod worker"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/clue-transcription.git
   git push -u origin main
   ```
3. In RunPod → **Serverless** → **New Endpoint** → **GitHub** as image source.
4. Select this repo, branch `main`, Dockerfile path `./Dockerfile`.
5. GPU: **16GB+ VRAM** (recommended 24GB). Container disk ≥ **20GB** (model baked in).
6. After build, copy **Endpoint ID** + **API key**.

## Test

```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"audio_url\":\"https://example.com/call.mp3\",\"language\":\"uk\"}}"
```

Or send `audio_base64` instead of `audio_url`.

## Env (optional)

| Variable | Default | Notes |
|----------|---------|--------|
| `WHISPER_MODEL` | `large-v3` | or `large-v3-turbo` |
| `WHISPER_DEVICE` | `cuda` | |
| `WHISPER_COMPUTE_TYPE` | `float16` | use `int8_float16` if VRAM tight |

Backend Clue wiring (RunPod client) comes after the endpoint is live.
