# Nemo ASR Endpoints And Setup

This document describes the split `nemo-asr` service running behind:

- local HTTP: `http://127.0.0.1:9090`
- local WebSocket: `ws://127.0.0.1:9090/ws`
- public HTTP: `https://asr.skeba.info`
- public WebSocket: `wss://asr.skeba.info/ws`

Internally it now consists of:

- Docker gateway on `9090`
- host-native NeMo worker on `9091`

It covers:

- service purpose
- endpoint behavior
- authentication
- Docker setup
- Cloudflare tunnel setup
- example requests

## Overview

`nemo-asr` is a dedicated speech-to-text service built around NeMo streaming ASR.
It also supports VAD and Sortformer speaker diarization.

It is intentionally separate from:

- `tts.skeba.info` for text-to-speech
- the LLM layer for text generation

That gives you a clean voice pipeline:

- ASR: speech to text
- LLM: text to text
- TTS: text to speech

## Local Service Files

Main service files:

- [gateway.py](C:/docker/voice/nemo-asr/gateway.py)
- [server.py](C:/docker/voice/nemo-asr/server.py)
- [start-host-asr.ps1](C:/docker/voice/nemo-asr/start-host-asr.ps1)
- [docker-compose.yaml](C:/docker/voice/nemo-asr/docker-compose.yaml)
- [Dockerfile](C:/docker/voice/nemo-asr/Dockerfile)
- [requirements.txt](C:/docker/voice/nemo-asr/requirements.txt)
- [gateway-requirements.txt](C:/docker/voice/nemo-asr/gateway-requirements.txt)
- [README.md](C:/docker/voice/nemo-asr/README.md)
- [test_client.py](C:/docker/voice/nemo-asr/test_client.py)
- [asr-websocket-react](C:/docker/voice/asr-websocket-react)

Tunnel files:

- [cloudflared config](C:/docker/cloudflared-asr/cloudflared/config.yml)
- [cloudflared compose](C:/docker/cloudflared-asr/docker-compose.yaml)

## Endpoints

### `GET /`

Purpose:

- basic endpoint inventory

Auth:

- no auth required

### `GET /health`

Purpose:

- health check
- readiness
- lifecycle visibility

Auth:

- no auth required

Example response:

```json
{
  "status": "idle",
  "model_loaded": false,
  "model": "C:\\docker\\nemo-asr\\models\\nemotron-speech-streaming-en-0.6b.nemo",
  "vad_model_loaded": false,
  "vad_model": "vad_multilingual_marblenet",
  "diarization_model_loaded": false,
  "diarization_model": "nvidia/diar_streaming_sortformer_4spk-v2",
  "right_context": 1,
  "sample_rate": 16000,
  "busy": false,
  "idle_unload_seconds": 60,
  "preload_model": false
}
```

Example:

```bash
curl https://asr.skeba.info/health
```

### `GET /api`

Purpose:

- returns API metadata
- lists routes
- shows auth and lifecycle info

Auth:

- required

Example:

```bash
curl https://asr.skeba.info/api \
  -H "Authorization: Bearer 123::your-api-key"
```

### `POST /api/transcribe`

Purpose:

- fetch a remote audio or video URL
- convert it to mono 16kHz audio
- run ASR, VAD, and optional speaker diarization

Auth:

- required

Request body:

```json
{
  "audio_url": "https://example.com/audio-or-video.mp4",
  "mode": "asr_vad",
  "enable_diarization": true,
  "vad": {
    "threshold": 0.5,
    "window_seconds": 0.63,
    "hop_seconds": 0.1,
    "min_speech_seconds": 0.2,
    "min_silence_seconds": 0.15
  }
}
```

Modes:

- `asr`: transcript only
- `vad`: VAD only
- `asr_vad`: transcript plus VAD output
- `enable_diarization: true`: include Sortformer speaker segments in the response

Example response:

```json
{
  "mode": "asr_vad",
  "text": "Hello from the ASR service",
  "audio_url": "https://example.com/audio-or-video.mp4",
  "duration_ms": 7168,
  "sample_rate": 16000,
  "model": "C:\\docker\\nemo-asr\\models\\nemotron-speech-streaming-en-0.6b.nemo",
  "diarization": {
    "speaker_segments": [
      {
        "start_ms": 0,
        "end_ms": 1520,
        "speaker": "speaker_0"
      },
      {
        "start_ms": 3120,
        "end_ms": 5670,
        "speaker": "speaker_1"
      }
    ],
    "speaker_count": 2,
    "diarization_model": "nvidia/diar_streaming_sortformer_4spk-v2"
  },
  "vad": {
    "speech_segments": [
      {
        "start_ms": 1200,
        "end_ms": 4120,
        "max_speech_probability": 0.93
      }
    ]
  }
}
```

Example curl:

```bash
curl -X POST "https://asr.skeba.info/api/transcribe" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"audio_url":"https://example.com/file.mp4","mode":"asr_vad","enable_diarization":true,"vad":{"threshold":0.5,"window_seconds":0.63,"hop_seconds":0.1}}'
```

### `POST /api/vad`

Purpose:

- fetch a remote audio or video URL
- convert it to mono 16kHz audio
- return only voice activity detection results

Auth:

- required

Example curl:

```bash
curl -X POST "https://asr.skeba.info/api/vad" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"audio_url":"https://example.com/file.mp4"}'
```

### `POST /api/diarize`

Purpose:

- fetch a remote audio or video URL
- convert it to mono 16kHz audio
- return only Sortformer speaker diarization segments

Auth:

- required

Example curl:

```bash
curl -X POST "https://asr.skeba.info/api/diarize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"audio_url":"https://example.com/file.mp4"}'
```

### `GET /ws`

Purpose:

- streaming websocket ASR
- accepts PCM16 mono 16kHz chunks
- can emit VAD turn events
- can attach Sortformer speaker labels to committed turns

Auth:

- required before websocket upgrade

Supported auth headers:

- `Authorization: Bearer <userId>::<apiKey>`
- `x-web-key: <userId>::<webKey>`

Message patterns:

- binary audio frames
- JSON messages such as:
  - `{"message_type":"input_audio_chunk","audio_base_64":"..."}`
  - `{"type":"reset","finalize":true}`
  - `{"type":"end","finalize":true}`

Server messages include:

- `session_updated`
- `partial_transcript`
- `committed_transcript`
- `speech_started`
- `speech_ended`
- `speaker_segment`
- `error`

WebSocket diarization:

- add `enable_diarization=1` in the query string, or send `{"type":"configure","enable_diarization":true}`
- diarization is turn-level in the live websocket path
- when VAD closes a turn, the committed transcript can include:
  - `speaker`
  - `speaker_segment`
  - `diarization`

Example:

```text
wss://asr.skeba.info/ws?mode=asr_vad&enable_diarization=1&emit_diarization_segments=1
```

## Authentication

Protected routes use the same MySQL-backed auth pattern as the Ollama stack and the TTS service.

Supported auth modes:

1. API clients

Use:

```http
Authorization: Bearer <userId>::<apiKey>
```

2. Browser or admin clients

Use:

```http
x-web-key: <userId>::<webKey>
```

Resolution order:

1. If `x-web-key` is present, verify it first
2. Otherwise, fall back to `Authorization: Bearer ...`

Backend verification URLs used by the service:

- API key check: `http://mysql_api:3001/db-api/backend-api/ping`
- Web key check: `http://mysql_api:3001/db-api/web-api/whoami`

Status convention:

- `401` for missing or invalid client credentials
- `502` if the auth backend cannot be reached or returns an unusable response

Open routes:

- `/`
- `/health`

Protected routes:

- `/api`
- `/api/transcribe`
- `/api/vad`
- `/ws`

## Docker Setup

### Compose

The service is started with:

```powershell
docker compose -f C:\docker\voice\nemo-asr\docker-compose.yaml up -d --build
```

Current important environment variables in [docker-compose.yaml](C:/docker/voice/nemo-asr/docker-compose.yaml):

- `HOST=0.0.0.0`
- `PORT=9090`
- `ASR_UPSTREAM_BASE=http://host.docker.internal:9091`
- `ASR_UPSTREAM_TIMEOUT=300`
- `REQUIRE_AUTH=1`
- `DB_API_BASE=http://mysql_api:3001`
- `DB_API_PREFIX=/db-api`
- `DB_API_TIMEOUT=8.0`

### Docker Networks

The service must be attached to `n8n_default` so it can resolve `mysql_api`.

That is already configured in [docker-compose.yaml](C:/docker/voice/nemo-asr/docker-compose.yaml).

### Host Worker

The actual NeMo inference process now runs outside Docker.

Start it with:

```powershell
C:\docker\voice\nemo-asr\start-host-asr.ps1
```

Default host-worker behavior:

- binds to `127.0.0.1:9091`
- uses [server.py](C:/docker/voice/nemo-asr/server.py)
- auth disabled locally with `REQUIRE_AUTH=0`
- cold-starts on first request
- unloads after `60` seconds idle
- checks for inactivity every `15` seconds

Default model path:

- `C:\docker\voice\nemo-asr\models\nemotron-speech-streaming-en-0.6b.nemo`

Default VAD model path/name:

- `vad_multilingual_marblenet`

Note:

- in this installed NeMo runtime, `vad_multilingual_frame_marblenet` is not an available pretrained checkpoint
- the supported multilingual MarbleNet VAD model is `vad_multilingual_marblenet`

The host worker is where CUDA, `ffmpeg`, and model loading now happen.

## Cloudflare Setup

The public hostname is:

- `asr.skeba.info`

The dedicated tunnel config lives in:

- [cloudflared-asr/cloudflared/config.yml](C:/docker/cloudflared-asr/cloudflared/config.yml)

Current tunnel route:

- `asr.skeba.info -> http://host.docker.internal:9090`

Tunnel startup:

```powershell
docker compose -f C:\docker\cloudflared-asr\docker-compose.yaml up -d
```

## Local Testing

### Health

```powershell
Invoke-RestMethod http://127.0.0.1:9090/health
```

### API info

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api `
  -Headers @{ Authorization = "Bearer 123::your-api-key" }
```

### Remote URL transcription

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/transcribe `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer 123::your-api-key" } `
  -Body '{"audio_url":"https://example.com/file.mp4","mode":"asr"}'
```

### VAD only

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/vad `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer 123::your-api-key" } `
  -Body '{"audio_url":"https://example.com/file.mp4"}'
```

### Combined ASR + VAD

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/transcribe `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer 123::your-api-key" } `
  -Body '{"audio_url":"https://example.com/file.mp4","mode":"asr_vad","vad":{"threshold":0.5,"window_seconds":0.63,"hop_seconds":0.1}}'
```

### Streaming websocket smoke test

```powershell
python C:\docker\voice\nemo-asr\test_client.py C:\path\to\sample.wav
```

If the client needs headers, pass the same bearer or web key used for the HTTP endpoints.

## Notes

- `/health` remains public so uptime checks and tunnel diagnostics keep working.
- Auth is enforced before websocket upgrade on `/ws`.
- API auth headers are validated at the gateway layer and are not part of the ASR model itself.
- Remote `/api/transcribe` and `/api/vad` use `ffmpeg` in the host-native worker, so MP4 and common audio formats can share the same endpoint family.
