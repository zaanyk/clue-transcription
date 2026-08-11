# clue-transcription

RunPod Serverless worker: **Whisper large-v3-turbo** (faster-whisper), tuned for speed.

## Deploy on RunPod (Docker Registry — recommended)

GitHub Import is unreliable (handler scanner). Use Docker instead:

1. Wait for GitHub Action **Build and push worker image** to finish (Actions tab).
2. Make the package public: GitHub → Packages → `clue-transcription` → Package settings → Change visibility → Public.
3. RunPod → Serverless → **New Endpoint** → **Import from Docker Registry** (NOT GitHub).
4. Image:
   ```
   ghcr.io/zaanyk/clue-transcription:latest
   ```
5. Type: **Queue**, GPU 16GB+ (better 24GB), disk ≥ 20GB → Deploy.
6. For stable latency set **Min workers = 1** (avoids cold start).

## Speed defaults

| Setting | Default |
|--------|---------|
| Model | `large-v3-turbo` |
| Beam size | `1` |
| VAD | on (skips silence) |
| `condition_on_previous_text` | `false` |

Override via job `input` or container env: `WHISPER_MODEL`, `WHISPER_BEAM_SIZE`, `WHISPER_COMPUTE_TYPE`.

## Local image (optional)

```bash
docker build --platform linux/amd64 -t ghcr.io/zaanyk/clue-transcription:latest .
docker push ghcr.io/zaanyk/clue-transcription:latest
```

## Request format

```json
{
  "input": {
    "audio_url": "https://example.com/call.mp3",
    "language": "uk",
    "model": "large-v3-turbo",
    "beam_size": 1
  }
}
```

Or `audio_base64` instead of `audio_url`. For quality-first runs, pass `"model": "large-v3"` and `"beam_size": 5`.
