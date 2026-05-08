# Pocket TTS service

This is a dedicated HTTP wrapper around [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).

The service now runs in a split layout:

- Docker container on `9080`: auth + HTTP gateway only
- host-native worker on `9081`: actual Pocket TTS model process

This keeps the public endpoint stable while moving model memory outside Docker.

## Endpoints

- `GET /health`
- `GET /api`
- `GET /api/voices`
- `POST /api/voices/clone`
- `DELETE /api/voices/clone/{clone_id}`
- `POST /api/synthesize`
- `POST /api/synthesize/stream`

## Auth

`/api`, `/api/voices`, and `/api/synthesize` require one of these headers:

- `Authorization: Bearer <userId>::<apiKey>`
- `x-web-key: <userId>::<webKey>`

Verification is delegated to the MySQL-backed auth API at `http://mysql_api:3001/db-api`.

## Start locally

```powershell
C:\docker\voice\pocket-tts\start-host-tts.ps1
docker compose -f C:\docker\voice\pocket-tts\docker-compose.yaml up -d --build
```

The host-native worker defaults to:

- `PRELOAD_MODEL=0`
- `IDLE_UNLOAD_SECONDS=300`
- `IDLE_CHECK_INTERVAL_SECONDS=30`

So the model cold-starts on demand and unloads after inactivity.

## Example

Return WAV directly:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:9080/api/synthesize `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer 123::your-api-key" } `
  -Body '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"wav"}' `
  -OutFile C:\docker\voice\pocket-tts\sample.wav
```

Return base64 JSON:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9080/api/synthesize `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer 123::your-api-key" } `
  -Body '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"base64"}'
```

`curl` example:

```bash
curl -X POST "https://tts.skeba.info/api/synthesize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS","voice":"alba","language":"english","response_format":"base64"}'
```

List voices and saved clones:

```bash
curl "https://tts.skeba.info/api/voices" \
  -H "Authorization: Bearer 123::your-api-key"
```

Create a saved clone:

```bash
curl -X POST "https://tts.skeba.info/api/voices/clone" \
  -H "Authorization: Bearer 123::your-api-key" \
  -F "audio=@/path/to/reference.wav" \
  -F "clone_id=my_voice" \
  -F "language=english"
```

Delete a saved clone:

```bash
curl -X DELETE "https://tts.skeba.info/api/voices/clone/my_voice" \
  -H "Authorization: Bearer 123::your-api-key"
```

Stream raw PCM progressively:

```bash
curl -N -X POST "https://tts.skeba.info/api/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS. This can stream while it is still generating.","voice":"alba","language":"english","stream_format":"pcm_s16le"}' \
  --output response.pcm
```

Stream a WAV-style response progressively:

```bash
curl -N -X POST "https://tts.skeba.info/api/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123::your-api-key" \
  -d '{"text":"Hello from Pocket TTS. This can stream while it is still generating.","voice":"alba","language":"english","stream_format":"wav"}' \
  --output response-stream.wav
```

## Notes

- Pocket TTS is CPU-oriented and intentionally separated from the GPU-backed ASR service.
- Docker validates auth and forwards requests to `http://host.docker.internal:9081`.
- The host-native worker caches the model and voice states in memory for responsiveness.
- The service can unload the model after inactivity with `IDLE_UNLOAD_SECONDS`.
- `/api/synthesize/stream` is the low-latency path for long reads. It supports `pcm_s16le` streaming and a streamed `wav` variant.
- Saved clones are exposed under `/api/voices/clone`, but actual clone creation requires Pocket TTS voice-cloning weights to be available in the host runtime.
