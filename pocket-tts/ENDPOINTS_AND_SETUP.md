# Pocket TTS Endpoints And Setup

This document describes the `pocket-tts` service running behind:

- local HTTP: `http://127.0.0.1:9080`
- public HTTP: `https://tts.skeba.info`

It covers:

- service purpose
- endpoint behavior
- authentication
- gateway and host worker setup
- Cloudflare tunnel setup
- example requests

## Overview

`pocket-tts` is a dedicated text-to-speech service built around [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).

It is intentionally separate from:

- `asr.skeba.info` for speech-to-text
- the LLM stack for text generation

That separation keeps the voice pipeline modular:

- ASR: speech to text
- LLM: text to text
- TTS: text to speech

## Local Service Files

Main service files:

- [app.py](C:/docker/voice/pocket-tts/app.py)
- [gateway.py](C:/docker/voice/pocket-tts/gateway.py)
- [start-host-tts.ps1](C:/docker/voice/pocket-tts/start-host-tts.ps1)
- [docker-compose.yaml](C:/docker/voice/pocket-tts/docker-compose.yaml)
- [Dockerfile](C:/docker/voice/pocket-tts/Dockerfile)
- [requirements.txt](C:/docker/voice/pocket-tts/requirements.txt)
- [gateway-requirements.txt](C:/docker/voice/pocket-tts/gateway-requirements.txt)
- [README.md](C:/docker/voice/pocket-tts/README.md)

Tunnel files:

- [cloudflared config](C:/docker/cloudflared-tts/cloudflared/config.yml)
- [cloudflared compose](C:/docker/cloudflared-tts/docker-compose.yaml)

## Endpoints

### `GET /health`

Purpose:

- health check
- lifecycle visibility
- safe for uptime checks

Auth:

- no auth required

Example response:

```json
{
  "status": "idle",
  "model_loaded": false,
  "language": null,
  "busy": false,
  "idle_unload_seconds": 300,
  "preload_model": false
}
```

Example:

```bash
curl https://tts.skeba.info/health
```

### `GET /api`

Purpose:

- returns API metadata
- lists routes
- shows supported languages and built-in voices

Auth:

- required

Example:

```bash
curl https://tts.skeba.info/api \
  -H "Authorization: Bearer 123::your-api-key"
```

### `GET /api/voices`

Purpose:

- lists available built-in voices
- lists supported languages
- lists saved clones
- reports whether voice cloning is currently available

Auth:

- required

Example:

```bash
curl https://tts.skeba.info/api/voices \
  -H "Authorization: Bearer 123::your-api-key"
```

### `POST /api/voices/clone`

Purpose:

- upload a reference audio file
- save a clone by ID for later synthesis

Auth:

- required

Multipart fields:

- `audio`: reference WAV/MP3/OGG file
- `clone_id`: frontend-friendly identifier for the saved clone
- `language`: Pocket TTS language, default `english`

Example:

```bash
curl -X POST "https://tts.skeba.info/api/voices/clone" \
  -H "Authorization: Bearer 123::your-api-key" \
  -F "audio=@/path/to/reference.wav" \
  -F "clone_id=my_voice" \
  -F "language=english"
```

Response:

```json
{
  "success": true,
  "clone_id": "my_voice",
  "language": "english",
  "source_filename": "reference.wav",
  "created_at": "2026-05-06T21:00:00Z"
}
```

If the current host runtime does not have Pocket TTS voice-cloning weights available, this endpoint returns `503`.

### `DELETE /api/voices/clone/{clone_id}`

Purpose:

- delete a saved clone

Auth:

- required

Example:

```bash
curl -X DELETE "https://tts.skeba.info/api/voices/clone/my_voice" \
  -H "Authorization: Bearer 123::your-api-key"
```

### `POST /api/synthesize`

Purpose:

- synthesize text into speech

Auth:

- required

Request body:

```json
{
  "text": "Hello from Pocket TTS",
  "voice": "alba",
  "language": "english",
  "response_format": "base64"
}
```

Fields:

- `text`: required text to synthesize
- `voice`: built-in voice name, default `alba`
- `language`: model language, default `english`
- `response_format`: `wav` or `base64`

Behavior:

- `base64` returns JSON with `audio_base64`
- `wav` returns raw `audio/wav`

Example JSON response:

```json
{
  "audio_base64": "...",
  "mime_type": "audio/wav",
  "sample_rate": 24000,
  "voice": "alba",
  "language": "english",
  "duration_ms": 828
}
```

Example curl for JSON:

```bash
curl -X POST "https://tts.skeba.info/api/synthesize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"base64"}'
```

Example curl for WAV:

```bash
curl -X POST "https://tts.skeba.info/api/synthesize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"wav"}' \
  --output response.wav
```

### `POST /api/synthesize/stream`

Purpose:

- stream audio chunks as they are generated
- reduce time-to-first-audio for long reads

Auth:

- required

Request body:

```json
{
  "text": "Hello from Pocket TTS. This can stream while it is still generating.",
  "voice": "alba",
  "language": "english",
  "stream_format": "pcm_s16le"
}
```

Fields:

- `text`: required text to synthesize
- `voice`: built-in voice name, default `alba`
- `language`: model language, default `english`
- `stream_format`: `pcm_s16le` or `wav`

Behavior:

- `pcm_s16le` returns raw 16-bit little-endian mono PCM progressively
- `wav` returns a streamed WAV header followed by PCM chunks

Important response headers:

- `X-TTS-Stream-Format`
- `X-Audio-Format`
- `X-Audio-Channels`
- `X-Audio-Sample-Rate`

Example curl for streaming PCM:

```bash
curl -N -X POST "https://tts.skeba.info/api/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS. This can stream while it is still generating.","voice":"alba","language":"english","stream_format":"pcm_s16le"}' \
  --output response.pcm
```

Example curl for streamed WAV:

```bash
curl -N -X POST "https://tts.skeba.info/api/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS. This can stream while it is still generating.","voice":"alba","language":"english","stream_format":"wav"}' \
  --output response-stream.wav
```

Example Python client for progressive reads:

```python
import requests

url = "https://tts.skeba.info/api/synthesize/stream"
headers = {"Authorization": "Bearer 123::your-api-key"}
payload = {
    "text": "Long text that should start playing before the whole response is finished.",
    "voice": "alba",
    "language": "english",
    "stream_format": "pcm_s16le",
}

with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as r:
    r.raise_for_status()
    sample_rate = r.headers.get("X-Audio-Sample-Rate", "24000")
    for chunk in r.iter_content(chunk_size=None):
        if chunk:
            print("got", len(chunk), "bytes at", sample_rate, "Hz")
```

## Authentication

Protected routes use the same MySQL-backed auth pattern as the Ollama stack.

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

Open route:

- `/health`

Protected routes:

- `/api`
- `/api/voices`
- `/api/voices/clone`
- `/api/synthesize`
- `/api/synthesize/stream`

## Runtime Layout

The public endpoint stays the same:

- `https://tts.skeba.info`

But the service is split into two layers:

- Docker gateway on `http://127.0.0.1:9080`
- host-native Pocket TTS worker on `http://127.0.0.1:9081`

Responsibilities:

- gateway: auth verification, request proxying, public-facing HTTP surface
- host worker: model loading, synthesis, idle unload lifecycle

The gateway forwards to:

- `http://host.docker.internal:9081`

That keeps model memory outside Docker while preserving the same external API.

## Gateway Setup

### Compose

Start the host worker first:

```powershell
C:\docker\voice\pocket-tts\start-host-tts.ps1
```

Then start the gateway container:

```powershell
docker compose -f C:\docker\voice\pocket-tts\docker-compose.yaml up -d --build
```

Current important environment variables in [docker-compose.yaml](C:/docker/voice/pocket-tts/docker-compose.yaml):

- `HOST=0.0.0.0`
- `PORT=9080`
- `TTS_UPSTREAM_BASE=http://host.docker.internal:9081`
- `TTS_UPSTREAM_TIMEOUT=300`
- `REQUIRE_AUTH=1`
- `DB_API_BASE=http://mysql_api:3001`
- `DB_API_PREFIX=/db-api`
- `DB_API_TIMEOUT=8.0`

### Docker Networks

The service must be attached to `n8n_default` so it can resolve `mysql_api`.

That is already configured in [docker-compose.yaml](C:/docker/voice/pocket-tts/docker-compose.yaml).

## Host Worker Setup

The host-native launcher is:

- [start-host-tts.ps1](C:/docker/voice/pocket-tts/start-host-tts.ps1)

It starts [app.py](C:/docker/voice/pocket-tts/app.py) on `127.0.0.1:9081` and disables auth on the host worker, since auth is enforced at the gateway layer.

Default host-worker behavior:

- no preload on startup
- unload after inactivity
- auto cold-start on the next synth request

Current defaults:

- preload disabled
- unload after `300` seconds idle
- check every `30` seconds

If you want to preload manually for testing:

```powershell
C:\docker\voice\pocket-tts\start-host-tts.ps1 -PreloadModel
```

## Cloudflare Setup

The public hostname is:

- `tts.skeba.info`

The dedicated tunnel config lives in:

- [cloudflared-tts/cloudflared/config.yml](C:/docker/cloudflared-tts/cloudflared/config.yml)

Current tunnel route:

- `tts.skeba.info -> http://host.docker.internal:9080`

Tunnel startup:

```powershell
docker compose -f C:\docker\cloudflared-tts\docker-compose.yaml up -d
```

## Supported Languages

Current language list exposed by the service:

- `english`
- `french`
- `german`
- `portuguese`
- `italian`
- `spanish`

## Built-In Voices

Current built-in voices exposed by the service:

- `alba`
- `anna`
- `azelma`
- `bill_boerst`
- `caro_davy`
- `charles`
- `cosette`
- `eponine`
- `eve`
- `fantine`
- `george`
- `jane`
- `jean`
- `javert`
- `marius`
- `mary`
- `michael`
- `paul`
- `peter_yearsley`
- `stuart_bell`
- `vera`

## Quick Checks

Health:

```bash
curl https://tts.skeba.info/health
```

Protected API info:

```bash
curl https://tts.skeba.info/api \
  -H "Authorization: Bearer 123::your-api-key"
```

Protected voice list:

```bash
curl https://tts.skeba.info/api/voices \
  -H "Authorization: Bearer 123::your-api-key"
```

Protected clone creation:

```bash
curl -X POST "https://tts.skeba.info/api/voices/clone" \
  -H "Authorization: Bearer 123::your-api-key" \
  -F "audio=@/path/to/reference.wav" \
  -F "clone_id=my_voice" \
  -F "language=english"
```

Protected synthesis:

```bash
curl -X POST "https://tts.skeba.info/api/synthesize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"base64"}'
```

Protected streamed synthesis:

```bash
curl -N -X POST "https://tts.skeba.info/api/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Long read that should stream back progressively.","voice":"alba","language":"english","stream_format":"pcm_s16le"}' \
  --output response.pcm
```

## Notes

- `/health` remains public so uptime checks and tunnel diagnostics keep working.
- API auth headers are validated at the gateway layer and are not forwarded to the model itself.
- Pocket TTS model memory now lives in the host-native worker rather than the Docker container.
- For low-latency playback, prefer `/api/synthesize/stream` over `/api/synthesize` for long text.
