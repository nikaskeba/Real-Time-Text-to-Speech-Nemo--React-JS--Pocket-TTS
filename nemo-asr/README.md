# Nemo ASR local service

This folder contains a split ASR setup for Nemotron / NeMo speech recognition:

- a lightweight Docker gateway on port `9090`
- a host-native NeMo worker on port `9091`

## Endpoints

- `GET /` returns a small endpoint inventory.
- `GET /health` returns service status and current model info.
- `GET /ws` accepts a streaming WebSocket connection for PCM16 audio, optional VAD, and optional speaker diarization.
- `GET /api` returns HTTP API details and example payloads.
- `POST /api/transcribe` accepts `audio_url`, `mode: asr | vad | asr_vad`, and optional `enable_diarization`.
- `POST /api/vad` accepts `audio_url` and returns VAD segments only.
- `POST /api/diarize` accepts `audio_url` and returns Sortformer speaker segments only.

## Local Docker test

1. Put your `.nemo` model file in `C:\docker\voice\nemo-asr\models\`.
2. Start the host-native worker:

```powershell
C:\docker\voice\nemo-asr\start-host-asr.ps1
```

3. Start the Docker gateway:

```powershell
docker compose -f C:\docker\voice\nemo-asr\docker-compose.yaml up -d --build
```

The React app served at `asr.skeba.info` lives beside this worker in
`C:\docker\voice\asr-websocket-react`; the ASR Dockerfile builds it from the
shared `C:\docker\voice` context.

4. Check gateway health:

```powershell
Invoke-RestMethod http://127.0.0.1:9090/health
```

5. Stream a local WAV file:

```powershell
python C:\docker\voice\nemo-asr\test_client.py C:\path\to\sample.wav
```

6. Inspect API info:

```powershell
Invoke-RestMethod http://127.0.0.1:9090/api
```

7. Transcribe a remote MP4 or audio URL:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/transcribe `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"audio_url":"https://example.com/file.mp4","mode":"asr"}'
```

8. Run VAD only:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/vad `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"audio_url":"https://example.com/file.mp4"}'
```

9. Run combined ASR + VAD:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/transcribe `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"audio_url":"https://example.com/file.mp4","mode":"asr_vad","vad":{"threshold":0.5,"window_seconds":0.63,"hop_seconds":0.1}}'
```

10. Run diarization only:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/diarize `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"audio_url":"https://example.com/file.mp4"}'
```

11. Run ASR + diarization:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:9090/api/transcribe `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"audio_url":"https://example.com/file.mp4","mode":"asr","enable_diarization":true}'
```

## Audio expectations

- 16 kHz
- mono
- 16-bit PCM WAV for the smoke test client

## Notes

- The Docker container is now only an auth/proxy gateway and does not run the NeMo model itself.
- The host-native worker uses [server.py](C:/docker/voice/nemo-asr/server.py) and binds to `127.0.0.1:9091` by default via [start-host-asr.ps1](C:/docker/voice/nemo-asr/start-host-asr.ps1).
- The default model path for the host worker is `C:\docker\voice\nemo-asr\models\nemotron-speech-streaming-en-0.6b.nemo`.
- Batch VAD now uses the NeMo checkpoint `vad_multilingual_marblenet`, which is the multilingual MarbleNet VAD model available in this runtime.
- Speaker diarization uses the NeMo Sortformer checkpoint `nvidia/diar_streaming_sortformer_4spk-v2`.
- WebSocket diarization is turn-level today: when `enable_diarization=1` and VAD closes a turn, the committed transcript includes `speaker` and `speaker_segment`, and the socket can emit `speaker_segment` events.
- Remote transcription still uses `ffmpeg`, but that work now happens in the host-native worker rather than inside Docker.
- Default lifecycle settings for the host worker: no preload at startup, unload after 60 idle seconds, check every 15 seconds.
