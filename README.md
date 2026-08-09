# clue-transcription

RunPod Serverless worker: **Whisper large-v3** via faster-whisper (CUDA).

## Entry point

- `handler.py` — required by RunPod GitHub indexer (`runpod.serverless.start`)
- Dockerfile CMD: `python -u /handler.py`

## Deploy

1. Repo: `https://github.com/zaanyk/clue-transcription`
2. RunPod → Serverless → New Endpoint → GitHub → this repo / `main` / `/Dockerfile`
3. Type: **Queue**
4. GPU: **16GB+** (recommended 24GB). Container disk ≥ **20GB**
5. After READY, copy Endpoint ID

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
