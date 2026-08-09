# clue-transcription

RunPod Serverless worker: **Whisper large-v3** (faster-whisper).

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
    "language": "uk"
  }
}
```

Or `audio_base64` instead of `audio_url`.
