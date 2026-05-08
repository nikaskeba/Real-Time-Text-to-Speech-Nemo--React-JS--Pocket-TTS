"""HTTP API wrapper for Kyutai Pocket TTS."""

import argparse
import asyncio
import base64
import contextlib
import gc
import io
import os
import json
import re
import shutil
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from pocket_tts import TTSModel
from pocket_tts.models.tts_model import export_model_state
from pydantic import BaseModel, Field


DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "9080"))
DEFAULT_PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "1") == "1"
DEFAULT_IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "1800"))
DEFAULT_IDLE_CHECK_INTERVAL_SECONDS = int(
    os.environ.get("IDLE_CHECK_INTERVAL_SECONDS", "30")
)

DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "english")
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "alba")
DEFAULT_RESPONSE_FORMAT = os.environ.get("TTS_DEFAULT_RESPONSE_FORMAT", "wav")
DEFAULT_CLONES_DIR = Path(os.environ.get("POCKET_TTS_CLONES_DIR", str(Path(__file__).resolve().parent / "voices")))
DB_API_BASE = os.environ.get("DB_API_BASE", "http://mysql_api:3001").rstrip("/")
DB_API_PREFIX = os.environ.get("DB_API_PREFIX", "/db-api")
DB_API_TIMEOUT = float(os.environ.get("DB_API_TIMEOUT", "8.0"))
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "1") == "1"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "https://app.skeba.info,https://admin.skeba.info,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

SUPPORTED_LANGUAGES = [
    "english",
    "french",
    "german",
    "portuguese",
    "italian",
    "spanish",
]

DEFAULT_VOICES = [
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "eve",
    "fantine",
    "george",
    "jane",
    "jean",
    "javert",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "stuart_bell",
    "vera",
]
CLONE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VoiceCloningUnavailableError(RuntimeError):
    pass


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(DEFAULT_VOICE, description="Built-in voice name or local/HF voice path")
    language: str = Field(DEFAULT_LANGUAGE, description="Pocket TTS language model name")
    response_format: str = Field(
        DEFAULT_RESPONSE_FORMAT,
        description="wav for binary audio, base64 for JSON-embedded audio",
    )


class SynthesizeStreamRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(DEFAULT_VOICE, description="Built-in voice name or local/HF voice path")
    language: str = Field(DEFAULT_LANGUAGE, description="Pocket TTS language model name")
    stream_format: str = Field(
        "pcm_s16le",
        description="pcm_s16le for raw streaming PCM, wav for a streamed WAV header plus PCM body",
    )


def _db_url(path: str) -> str:
    base = DB_API_BASE + (DB_API_PREFIX if DB_API_PREFIX else "")
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _parse_compound_key(value: str) -> tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None
    raw = value.strip()
    if "::" not in raw:
        return None, None
    user_id, secret = raw.split("::", 1)
    user_id = user_id.strip()
    secret = secret.strip()
    if not user_id or not secret:
        return None, None
    return user_id, secret


def _verify_api_bearer(auth_header: str, user_email: Optional[str] = None) -> dict[str, Any]:
    auth = (auth_header or "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="missing Authorization: Bearer <userId>::<apiKey>",
        )

    token = auth.split(" ", 1)[1].strip()
    user_id, api_key = _parse_compound_key(token)
    if not user_id or not api_key:
        raise HTTPException(
            status_code=401,
            detail="invalid bearer token format; expected userId::apiKey",
        )

    headers = {"x-api-key": token}
    email = (user_email or "").strip().lower()
    if email:
        headers["x-user-email"] = email

    try:
        resp = requests.get(
            _db_url("/backend-api/ping"),
            headers=headers,
            timeout=DB_API_TIMEOUT,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="auth verification failed") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid api credentials")

    try:
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="auth verification invalid response",
        ) from exc

    if isinstance(data, dict) and "user_id" not in data:
        data["user_id"] = data.get("userId") or user_id
    return data


def _verify_web_key(web_key: str) -> dict[str, Any]:
    web_key = (web_key or "").strip()
    if not web_key:
        raise HTTPException(status_code=401, detail="missing x-web-key")

    user_id, secret = _parse_compound_key(web_key)
    if not user_id or not secret:
        raise HTTPException(
            status_code=401,
            detail="invalid x-web-key format; expected userId::webKey",
        )

    try:
        resp = requests.get(
            _db_url("/web-api/whoami"),
            headers={"x-web-key": web_key},
            timeout=DB_API_TIMEOUT,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="auth verification failed") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid web key")

    try:
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="auth verification invalid response",
        ) from exc

    if isinstance(data, dict) and "user_id" not in data:
        data["user_id"] = user_id
    return data


async def require_user(request: Request) -> dict[str, Any]:
    if not REQUIRE_AUTH:
        return {"auth_bypassed": True}

    web_key = (request.headers.get("x-web-key") or "").strip()
    if web_key:
        return _verify_web_key(web_key)

    return _verify_api_bearer(
        request.headers.get("Authorization", ""),
        request.headers.get("x-user-email"),
    )


def _pcm16_bytes_from_audio(audio: Any) -> bytes:
    audio_np = audio.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (audio_np * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _streaming_wav_header(sample_rate: int, channels: int = 1, sample_width_bytes: int = 2) -> bytes:
    # Streaming WAV uses placeholder sizes because the final byte length is unknown up front.
    byte_rate = sample_rate * channels * sample_width_bytes
    block_align = channels * sample_width_bytes
    placeholder_size = 0xFFFFFFFF
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", placeholder_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, sample_width_bytes * 8),
            b"data",
            struct.pack("<I", placeholder_size),
        ]
    )


def _sanitize_clone_id(value: str) -> str:
    clone_id = (value or "").strip()
    if not clone_id:
        raise ValueError("clone_id is required")
    if not CLONE_ID_PATTERN.fullmatch(clone_id):
        raise ValueError("clone_id must match ^[A-Za-z0-9_-]{1,64}$")
    return clone_id


def _content_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix else ".bin"


class PocketTTSService:
    def __init__(
        self,
        preload_model: bool = DEFAULT_PRELOAD_MODEL,
        idle_unload_seconds: int = DEFAULT_IDLE_UNLOAD_SECONDS,
        idle_check_interval_seconds: int = DEFAULT_IDLE_CHECK_INTERVAL_SECONDS,
    ):
        self.preload_model = preload_model
        self.idle_unload_seconds = max(0, idle_unload_seconds)
        self.idle_check_interval_seconds = max(5, idle_check_interval_seconds)
        self.model: Optional[Any] = None
        self.model_language: Optional[str] = None
        self.voice_cache: dict[str, Any] = {}
        self.model_lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()
        self.active_requests = 0
        self.last_activity_at = time.monotonic()
        self.monitor_task: Optional[asyncio.Task] = None
        self.clones_dir = DEFAULT_CLONES_DIR
        self.clones_dir.mkdir(parents=True, exist_ok=True)

    def _mark_activity(self):
        self.last_activity_at = time.monotonic()

    def _is_busy(self) -> bool:
        return self.active_requests > 0

    def _normalize_language(self, language: str) -> str:
        value = (language or DEFAULT_LANGUAGE).strip().lower()
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Expected one of: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        return value

    def _clone_dir(self, clone_id: str) -> Path:
        return self.clones_dir / clone_id

    def _clone_voice_state_path(self, clone_id: str) -> Path:
        return self._clone_dir(clone_id) / "voice.safetensors"

    def _clone_metadata_path(self, clone_id: str) -> Path:
        return self._clone_dir(clone_id) / "metadata.json"

    def _clone_reference_path(self, clone_id: str, source_filename: str) -> Path:
        return self._clone_dir(clone_id) / f"reference{_content_suffix(source_filename)}"

    def _read_clone_metadata(self, clone_id: str) -> dict[str, Any]:
        metadata_path = self._clone_metadata_path(clone_id)
        if not metadata_path.exists():
            return {"clone_id": clone_id}
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_clone_ids(self) -> list[str]:
        if not self.clones_dir.exists():
            return []
        clone_ids: list[str] = []
        for entry in self.clones_dir.iterdir():
            if not entry.is_dir():
                continue
            if self._clone_voice_state_path(entry.name).exists():
                clone_ids.append(entry.name)
        return sorted(clone_ids)

    def list_clone_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for clone_id in self.list_clone_ids():
            metadata = self._read_clone_metadata(clone_id)
            summaries.append(
                {
                    "clone_id": clone_id,
                    "language": metadata.get("language"),
                    "source_filename": metadata.get("source_filename"),
                    "created_at": metadata.get("created_at"),
                }
            )
        return summaries

    def _resolve_voice_reference(self, voice: str) -> str:
        voice_name = (voice or DEFAULT_VOICE).strip()
        clone_path = self._clone_voice_state_path(voice_name)
        if clone_path.exists():
            return str(clone_path)
        return voice_name

    def _load_model_sync(self, language: str):
        logger.info("Loading Pocket TTS model for language '{}'", language)
        try:
            self.model = TTSModel.load_model(language=language)
        except TypeError:
            if language != "english":
                raise RuntimeError(
                    "This Pocket TTS build does not expose language-specific Python loading yet. "
                    "Use 'english' for now or switch to the upstream serve command."
                )
            self.model = TTSModel.load_model()
        self.model_language = language
        self.voice_cache = {}
        self._mark_activity()

    async def ensure_model_loaded(self, language: str):
        language = self._normalize_language(language)
        if self.model is not None and self.model_language == language:
            self._mark_activity()
            return

        async with self.model_lock:
            if self.model is not None and self.model_language == language:
                self._mark_activity()
                return
            await asyncio.get_event_loop().run_in_executor(None, self._load_model_sync, language)

    def _unload_model_sync(self):
        if self.model is None:
            return
        logger.info(
            "Unloading Pocket TTS model after {:.0f}s of inactivity",
            time.monotonic() - self.last_activity_at,
        )
        model = self.model
        self.model = None
        self.model_language = None
        self.voice_cache = {}
        del model
        gc.collect()

    async def idle_monitor(self):
        logger.info(
            "Pocket TTS idle monitor active: unload after {}s",
            self.idle_unload_seconds,
        )
        while True:
            await asyncio.sleep(self.idle_check_interval_seconds)
            if self.idle_unload_seconds <= 0 or self.model is None or self._is_busy():
                continue
            idle_for = time.monotonic() - self.last_activity_at
            if idle_for < self.idle_unload_seconds:
                continue
            async with self.model_lock:
                idle_for = time.monotonic() - self.last_activity_at
                if self.model is not None and not self._is_busy() and idle_for >= self.idle_unload_seconds:
                    await asyncio.get_event_loop().run_in_executor(None, self._unload_model_sync)

    async def startup(self):
        if self.preload_model:
            await self.ensure_model_loaded(DEFAULT_LANGUAGE)
        self.monitor_task = asyncio.create_task(self.idle_monitor())

    async def shutdown(self):
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task

    def _get_voice_state_sync(self, voice: str):
        resolved_voice = self._resolve_voice_reference(voice)
        if resolved_voice not in self.voice_cache:
            self.voice_cache[resolved_voice] = self.model.get_state_for_audio_prompt(resolved_voice)
        return self.voice_cache[resolved_voice]

    def _create_voice_clone_sync(
        self,
        clone_id: str,
        language: str,
        uploaded_path: Path,
        source_filename: str,
    ) -> dict[str, Any]:
        clone_dir = self._clone_dir(clone_id)
        if clone_dir.exists():
            raise FileExistsError(f"clone '{clone_id}' already exists")

        clone_dir.mkdir(parents=True, exist_ok=False)
        try:
            if not getattr(self.model, "has_voice_cloning", True):
                raise VoiceCloningUnavailableError(
                    "Voice cloning is not available in the current Pocket TTS model weights. "
                    "Accept the Hugging Face terms for kyutai/pocket-tts and authenticate locally, "
                    "then restart the host TTS worker."
                )

            reference_path = self._clone_reference_path(clone_id, source_filename)
            shutil.copyfile(uploaded_path, reference_path)

            voice_state = self.model.get_state_for_audio_prompt(str(reference_path), truncate=True)
            voice_state_path = self._clone_voice_state_path(clone_id)
            export_model_state(voice_state, voice_state_path)

            metadata = {
                "clone_id": clone_id,
                "language": language,
                "source_filename": source_filename,
                "reference_audio_filename": reference_path.name,
                "voice_state_filename": voice_state_path.name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            metadata_path = self._clone_metadata_path(clone_id)
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            self.voice_cache.pop(str(voice_state_path), None)
            self._mark_activity()
            return metadata
        except Exception:
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise

    async def create_voice_clone(
        self,
        clone_id: str,
        language: str,
        uploaded_path: Path,
        source_filename: str,
    ) -> dict[str, Any]:
        language = self._normalize_language(language)
        await self.ensure_model_loaded(language)
        self.active_requests += 1
        self._mark_activity()
        try:
            async with self.inference_lock:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._create_voice_clone_sync,
                    clone_id,
                    language,
                    uploaded_path,
                    source_filename,
                )
        finally:
            self.active_requests = max(0, self.active_requests - 1)
            self._mark_activity()

    def delete_voice_clone(self, clone_id: str) -> bool:
        clone_dir = self._clone_dir(clone_id)
        if not clone_dir.exists():
            return False

        voice_state_path = self._clone_voice_state_path(clone_id)
        self.voice_cache.pop(str(voice_state_path), None)
        shutil.rmtree(clone_dir)
        self._mark_activity()
        return True

    def _synthesize_sync(self, text: str, voice: str) -> tuple[bytes, int]:
        voice_state = self._get_voice_state_sync(voice)
        audio = self.model.generate_audio(voice_state, text)
        pcm_bytes = _pcm16_bytes_from_audio(audio)

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(self.model.sample_rate))
            wav_file.writeframes(pcm_bytes)
        return output.getvalue(), int(self.model.sample_rate)

    async def synthesize(self, request: SynthesizeRequest) -> tuple[bytes, int]:
        await self.ensure_model_loaded(request.language)
        self.active_requests += 1
        self._mark_activity()
        try:
            async with self.inference_lock:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._synthesize_sync,
                    request.text,
                    request.voice,
                )
        finally:
            self.active_requests = max(0, self.active_requests - 1)
            self._mark_activity()

    async def stream_synthesize(
        self,
        request: SynthesizeStreamRequest,
    ) -> tuple[int, Any]:
        await self.ensure_model_loaded(request.language)
        await self.inference_lock.acquire()
        self.active_requests += 1
        self._mark_activity()

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def worker():
            try:
                voice_state = self._get_voice_state_sync(request.voice)
                for audio_chunk in self.model.generate_audio_stream(voice_state, request.text):
                    loop.call_soon_threadsafe(
                        chunk_queue.put_nowait,
                        ("chunk", _pcm16_bytes_from_audio(audio_chunk)),
                    )
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", exc))

        threading.Thread(target=worker, daemon=True).start()
        sample_rate = int(self.model.sample_rate)

        async def iterator():
            try:
                if request.stream_format == "wav":
                    yield _streaming_wav_header(sample_rate)

                while True:
                    kind, payload = await chunk_queue.get()
                    if kind == "chunk":
                        yield payload
                        self._mark_activity()
                        continue
                    if kind == "done":
                        break
                    raise payload
            finally:
                self.active_requests = max(0, self.active_requests - 1)
                self._mark_activity()
                self.inference_lock.release()

        return sample_rate, iterator()

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.model is not None else "idle",
            "model_loaded": self.model is not None,
            "language": self.model_language,
            "busy": self._is_busy(),
            "idle_unload_seconds": self.idle_unload_seconds,
            "preload_model": self.preload_model,
        }

    def api_info(self) -> dict[str, Any]:
        return {
            "service": "pocket-tts",
            "default_language": DEFAULT_LANGUAGE,
            "default_voice": DEFAULT_VOICE,
            "auth": {
                "required": REQUIRE_AUTH,
                "modes": [
                    "Authorization: Bearer <userId>::<apiKey>",
                    "x-web-key: <userId>::<webKey>",
                ],
            },
            "supported_languages": SUPPORTED_LANGUAGES,
            "builtin_voices": DEFAULT_VOICES,
            "clones_dir": str(self.clones_dir),
            "voice_cloning_available": bool(self.model is not None and getattr(self.model, "has_voice_cloning", False)),
            "lifecycle": self.health(),
            "routes": {
                "health": {"method": "GET", "path": "/health"},
                "api": {"method": "GET", "path": "/api"},
                "voices": {"method": "GET", "path": "/api/voices"},
                "clone_voice": {
                    "method": "POST",
                    "path": "/api/voices/clone",
                    "multipart_form": {
                        "audio": "@/path/to/reference.wav",
                        "clone_id": "my_voice",
                        "language": DEFAULT_LANGUAGE,
                    },
                },
                "delete_clone": {
                    "method": "DELETE",
                    "path": "/api/voices/clone/{clone_id}",
                },
                "synthesize": {
                    "method": "POST",
                    "path": "/api/synthesize",
                    "body": {
                        "text": "Hello world",
                        "voice": DEFAULT_VOICE,
                        "language": DEFAULT_LANGUAGE,
                        "response_format": "wav",
                    },
                },
                "synthesize_stream": {
                    "method": "POST",
                    "path": "/api/synthesize/stream",
                    "body": {
                        "text": "Hello world",
                        "voice": DEFAULT_VOICE,
                        "language": DEFAULT_LANGUAGE,
                        "stream_format": "pcm_s16le",
                    },
                },
            },
        }


service = PocketTTSService()
app = FastAPI(title="Pocket TTS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS if CORS_ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "x-web-key", "x-user-email"],
)


@app.on_event("startup")
async def on_startup():
    await service.startup()


@app.on_event("shutdown")
async def on_shutdown():
    await service.shutdown()


@app.get("/")
async def root():
    return {"service": "pocket-tts", "status": "ok", "endpoints": ["/health", "/api"]}


@app.get("/health")
async def health():
    return service.health()


@app.get("/api")
async def api_info(_user: dict[str, Any] = Depends(require_user)):
    return service.api_info()


@app.get("/api/voices")
async def api_voices(_user: dict[str, Any] = Depends(require_user)):
    return {
        "default_voice": DEFAULT_VOICE,
        "builtin_voices": DEFAULT_VOICES,
        "clones": service.list_clone_summaries(),
        "voice_cloning_available": bool(service.model is not None and getattr(service.model, "has_voice_cloning", False)),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/api/voices/clone")
async def api_clone_voice(
    audio: UploadFile = File(...),
    clone_id: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    _user: dict[str, Any] = Depends(require_user),
):
    try:
        clone_id = _sanitize_clone_id(clone_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = _content_suffix(audio.filename or "")
    temp_path = DEFAULT_CLONES_DIR / f".upload-{clone_id}{suffix}"
    try:
        with temp_path.open("wb") as handle:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        try:
            metadata = await service.create_voice_clone(
                clone_id=clone_id,
                language=language,
                uploaded_path=temp_path,
                source_filename=audio.filename or f"{clone_id}{suffix}",
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VoiceCloningUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Pocket TTS voice clone creation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "success": True,
            "clone_id": clone_id,
            "language": metadata["language"],
            "source_filename": metadata["source_filename"],
            "created_at": metadata["created_at"],
        }
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        await audio.close()


@app.delete("/api/voices/clone/{clone_id}")
async def api_delete_clone_voice(
    clone_id: str,
    _user: dict[str, Any] = Depends(require_user),
):
    try:
        clone_id = _sanitize_clone_id(clone_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    deleted = service.delete_voice_clone(clone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"clone '{clone_id}' not found")
    return {"success": True, "clone_id": clone_id}


@app.post("/api/synthesize")
async def api_synthesize(
    request: SynthesizeRequest,
    _user: dict[str, Any] = Depends(require_user),
):
    response_format = (request.response_format or "wav").strip().lower()
    if response_format not in {"wav", "base64"}:
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'base64'")

    started = time.perf_counter()
    try:
        wav_bytes, sample_rate = await service.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pocket TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = int((time.perf_counter() - started) * 1000)
    if response_format == "base64":
        return JSONResponse(
            {
                "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
                "mime_type": "audio/wav",
                "sample_rate": sample_rate,
                "voice": request.voice,
                "language": request.language,
                "duration_ms": duration_ms,
            }
        )

    headers = {
        "X-TTS-Voice": request.voice,
        "X-TTS-Language": request.language,
        "X-TTS-Duration-Ms": str(duration_ms),
        "X-TTS-Sample-Rate": str(sample_rate),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/api/synthesize/stream")
async def api_synthesize_stream(
    request: SynthesizeStreamRequest,
    _user: dict[str, Any] = Depends(require_user),
):
    stream_format = (request.stream_format or "pcm_s16le").strip().lower()
    if stream_format not in {"pcm_s16le", "wav"}:
        raise HTTPException(status_code=400, detail="stream_format must be 'pcm_s16le' or 'wav'")

    request.stream_format = stream_format
    try:
        sample_rate, chunk_iterator = await service.stream_synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pocket TTS streaming synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "audio/wav" if stream_format == "wav" else "application/octet-stream"
    headers = {
        "X-TTS-Voice": request.voice,
        "X-TTS-Language": request.language,
        "X-TTS-Sample-Rate": str(sample_rate),
        "X-TTS-Stream-Format": stream_format,
        "X-Audio-Format": "pcm_s16le",
        "X-Audio-Channels": "1",
        "X-Audio-Sample-Rate": str(sample_rate),
        "Cache-Control": "no-store",
    }
    return StreamingResponse(chunk_iterator, media_type=media_type, headers=headers)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Pocket TTS HTTP API")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
