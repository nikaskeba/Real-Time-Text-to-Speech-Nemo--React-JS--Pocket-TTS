# Voice Stack

Local voice package for real-time ASR, TTS, and a React chat interface.

## Projects

- `nemo-asr`: Host-native NeMo/Nemotron ASR worker plus Docker web gateway.
- `pocket-tts`: Host-native Pocket TTS worker plus Docker API gateway.
- `asr-websocket-react`: React frontend served by the ASR web gateway.

## What Is Not Committed

Model files, cloned voices, virtual environments, build outputs, logs, generated audio, and local secrets are intentionally ignored. Users should download their own model weights and create their own `.env` files from the examples.

## Quick Start

1. Copy `nemo-asr/.env.example` to `nemo-asr/.env` and add your local keys/passwords.
2. Put the ASR `.nemo` model in `nemo-asr/models/`.
3. Start the host workers:

```powershell
C:\docker\voice\nemo-asr\start-host-asr.ps1
C:\docker\voice\pocket-tts\start-host-tts.ps1
```

4. Start the Docker gateways:

```powershell
docker compose -f C:\docker\voice\nemo-asr\docker-compose.yaml up -d --build
docker compose -f C:\docker\voice\pocket-tts\docker-compose.yaml up -d --build
```

## Public Deployment Notes

This repository is intended to be publishable without private weights or credentials. Before publishing, verify `git status --ignored` and confirm `.env`, model files, voices, logs, and virtual environments are ignored.
