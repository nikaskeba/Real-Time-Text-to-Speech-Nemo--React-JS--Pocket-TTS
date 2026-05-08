"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import argparse
import asyncio
import base64
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import shutil
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from loguru import logger
from omegaconf import OmegaConf

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


# Default model - HuggingFace model name (auto-downloads) or local .nemo path
DEFAULT_MODEL = os.environ.get(
    "NEMO_MODEL",
    "models/nemotron-speech-streaming-en-0.6b.nemo",
)
DEFAULT_IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "900"))
DEFAULT_IDLE_CHECK_INTERVAL_SECONDS = int(
    os.environ.get("IDLE_CHECK_INTERVAL_SECONDS", "30")
)
DEFAULT_PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "1") == "1"
DEFAULT_VAD_MODEL = os.environ.get("NEMO_VAD_MODEL", "vad_multilingual_marblenet")
DEFAULT_DIARIZATION_MODEL = os.environ.get(
    "NEMO_DIARIZATION_MODEL",
    "nvidia/diar_streaming_sortformer_4spk-v2",
)
DEFAULT_MAX_REQUEST_SIZE_MB = float(os.environ.get("ASR_MAX_REQUEST_SIZE_MB", "50"))
DEFAULT_VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
DEFAULT_VAD_WINDOW_SECONDS = float(os.environ.get("VAD_WINDOW_SECONDS", "0.63"))
DEFAULT_VAD_HOP_SECONDS = float(os.environ.get("VAD_HOP_SECONDS", "0.1"))
DEFAULT_VAD_MIN_SPEECH_SECONDS = float(os.environ.get("VAD_MIN_SPEECH_SECONDS", "0.2"))
DEFAULT_VAD_MIN_SILENCE_SECONDS = float(os.environ.get("VAD_MIN_SILENCE_SECONDS", "0.15"))
DB_API_BASE = os.environ.get("DB_API_BASE", "http://mysql_api:3001").rstrip("/")
DB_API_PREFIX = os.environ.get("DB_API_PREFIX", "/db-api")
DB_API_TIMEOUT_SECONDS = float(os.environ.get("DB_API_TIMEOUT", "8.0"))
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "1") == "1"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "https://app.skeba.info,https://admin.skeba.info,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
)
CORS_ALLOWED_HEADERS = "Authorization, Content-Type, x-web-key, x-user-email"
CORS_ALLOWED_METHODS = "GET, POST, DELETE, OPTIONS"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""

    id: str
    websocket: Any

    accumulated_audio: Optional[np.ndarray] = None
    emitted_frames: int = 0
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None
    previous_hypotheses: Any = None
    pred_out_stream: Any = None
    current_text: str = ""
    last_emitted_text: str = ""
    overlap_buffer: Optional[np.ndarray] = None
    websocket_mode: str = "asr"
    emit_vad_frames: bool = False
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_window_seconds: float = DEFAULT_VAD_WINDOW_SECONDS
    vad_hop_seconds: float = DEFAULT_VAD_HOP_SECONDS
    vad_min_speech_seconds: float = DEFAULT_VAD_MIN_SPEECH_SECONDS
    vad_min_silence_seconds: float = DEFAULT_VAD_MIN_SILENCE_SECONDS
    auto_finalize_on_vad_end: bool = True
    diarization_enabled: bool = False
    emit_diarization_segments: bool = True
    diarization_audio: Optional[np.ndarray] = None
    diarization_last_emitted_end_ms: int = 0
    total_samples_received: int = 0
    accumulated_audio_base_samples: int = 0
    vad_processed_samples: int = 0
    vad_speech_active: bool = False
    vad_active_start_ms: Optional[int] = None
    vad_candidate_start_ms: Optional[int] = None
    vad_candidate_last_speech_end_ms: Optional[int] = None


class ASRServer:
    """WebSocket server for streaming ASR with true incremental processing."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 9090,
        right_context: int = 1,
        idle_unload_seconds: int = DEFAULT_IDLE_UNLOAD_SECONDS,
        idle_check_interval_seconds: int = DEFAULT_IDLE_CHECK_INTERVAL_SECONDS,
        preload_model: bool = DEFAULT_PRELOAD_MODEL,
        vad_model: str = DEFAULT_VAD_MODEL,
        diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    ):
        self.model_name_or_path = model
        self.host = host
        self.port = port
        self.right_context = right_context
        self.idle_unload_seconds = max(0, idle_unload_seconds)
        self.idle_check_interval_seconds = max(5, idle_check_interval_seconds)
        self.preload_model = preload_model
        self.vad_model_name = vad_model
        self.diarization_model_name = diarization_model
        self.model = None
        self.vad_model = None
        self.diarization_model = None
        self.vad_speech_label_index = 1
        self.sample_rate = 16000
        self.inference_lock = asyncio.Lock()
        self.sessions: dict[str, ASRSession] = {}
        self.active_http_requests = 0
        self.last_activity_at = time.monotonic()
        self.idle_monitor_task: Optional[asyncio.Task] = None
        self.model_loaded = False
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None
        self.drop_extra = 0
        self.final_padding_frames = 0
        self.overlap_samples = None

    def _cors_origin_for_request(self, request: web.Request) -> Optional[str]:
        origin = (request.headers.get("Origin") or "").strip()
        if not origin:
            return None
        if "*" in CORS_ALLOWED_ORIGINS:
            return origin
        return origin if origin in CORS_ALLOWED_ORIGINS else None

    def _apply_cors_headers(self, request: web.Request, response: web.StreamResponse) -> web.StreamResponse:
        origin = self._cors_origin_for_request(request)
        if not origin:
            return response

        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = CORS_ALLOWED_HEADERS
        response.headers["Access-Control-Allow-Methods"] = CORS_ALLOWED_METHODS
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
        return response

    @web.middleware
    async def cors_middleware(self, request: web.Request, handler):
        if request.method == "OPTIONS":
            response = web.Response(status=204)
            return self._apply_cors_headers(request, response)

        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc

        return self._apply_cors_headers(request, response)

    def _mark_activity(self):
        """Record the last time the service did real work."""
        self.last_activity_at = time.monotonic()

    def _is_busy(self) -> bool:
        """Return True while live sessions or HTTP transcription jobs are active."""
        return bool(self.sessions) or self.active_http_requests > 0

    def _build_api_description(self) -> dict[str, Any]:
        """Describe the HTTP and websocket surfaces exposed by this service."""
        return {
            "service": "nemo-asr",
            "model_loaded": self.model_loaded,
            "sample_rate": self.sample_rate,
            "model": self.model_name_or_path,
            "vad_model": self.vad_model_name,
            "vad_model_loaded": self.vad_model is not None,
            "diarization_model": self.diarization_model_name,
            "diarization_model_loaded": self.diarization_model is not None,
            "auth": {
                "required": REQUIRE_AUTH,
                "modes": [
                    "Authorization: Bearer <userId>::<apiKey>",
                    "x-web-key: <userId>::<webKey>",
                ],
            },
            "lifecycle": {
                "preload_model": self.preload_model,
                "idle_unload_seconds": self.idle_unload_seconds,
                "idle_check_interval_seconds": self.idle_check_interval_seconds,
                "busy": self._is_busy(),
            },
            "routes": {
                "root": {
                    "method": "GET",
                    "path": "/",
                    "description": "Basic endpoint inventory",
                },
                "health": {
                    "method": "GET",
                    "path": "/health",
                    "description": "Readiness and model status",
                },
                "websocket": {
                    "method": "GET",
                    "path": "/ws",
                    "description": "Streaming websocket for PCM16 mono 16kHz chunks with optional VAD and speaker diarization events",
                    "query_params": {
                        "mode": "asr | vad | asr_vad",
                        "emit_vad_frames": "0 | 1",
                        "auto_finalize_on_vad_end": "0 | 1",
                        "enable_diarization": "0 | 1",
                        "emit_diarization_segments": "0 | 1",
                        "vad_threshold": DEFAULT_VAD_THRESHOLD,
                        "vad_window_seconds": DEFAULT_VAD_WINDOW_SECONDS,
                        "vad_hop_seconds": DEFAULT_VAD_HOP_SECONDS,
                    },
                    "configure_message": {
                        "type": "configure",
                        "mode": "asr_vad",
                        "emit_vad_frames": False,
                        "auto_finalize_on_vad_end": True,
                        "enable_diarization": True,
                        "emit_diarization_segments": True,
                        "vad": {
                            "threshold": DEFAULT_VAD_THRESHOLD,
                            "window_seconds": DEFAULT_VAD_WINDOW_SECONDS,
                            "hop_seconds": DEFAULT_VAD_HOP_SECONDS,
                            "min_speech_seconds": DEFAULT_VAD_MIN_SPEECH_SECONDS,
                            "min_silence_seconds": DEFAULT_VAD_MIN_SILENCE_SECONDS,
                        },
                    },
                },
                "api_info": {
                    "method": "GET",
                    "path": "/api",
                    "description": "HTTP API inventory and request examples",
                },
                "api_transcribe": {
                    "method": "POST",
                    "path": "/api/transcribe",
                    "description": "Fetch a remote audio/video URL, transcode to 16kHz mono, and run ASR, VAD, and optional speaker diarization",
                    "json_body": {
                        "audio_url": "https://example.com/audio-or-video.mp4",
                        "mode": "asr_vad",
                        "enable_diarization": True,
                        "vad": {
                            "threshold": DEFAULT_VAD_THRESHOLD,
                            "window_seconds": DEFAULT_VAD_WINDOW_SECONDS,
                            "hop_seconds": DEFAULT_VAD_HOP_SECONDS,
                        },
                    },
                },
                "api_vad": {
                    "method": "POST",
                    "path": "/api/vad",
                    "description": "Fetch a remote audio/video URL, transcode to 16kHz mono, and return voice activity segments",
                    "json_body": {
                        "audio_url": "https://example.com/audio-or-video.mp4",
                        "vad": {
                            "threshold": DEFAULT_VAD_THRESHOLD,
                            "window_seconds": DEFAULT_VAD_WINDOW_SECONDS,
                            "hop_seconds": DEFAULT_VAD_HOP_SECONDS,
                        },
                    },
                },
                "api_diarize": {
                    "method": "POST",
                    "path": "/api/diarize",
                    "description": "Fetch a remote audio/video URL, transcode to 16kHz mono, and return Sortformer speaker diarization segments",
                    "json_body": {
                        "audio_url": "https://example.com/audio-or-video.mp4",
                        "enable_diarization": True,
                    },
                },
            },
        }

    def _db_url(self, path: str) -> str:
        base = DB_API_BASE + (DB_API_PREFIX if DB_API_PREFIX else "")
        return f"{base.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
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

    async def _verify_api_bearer(
        self,
        auth_header: str,
        user_email: Optional[str] = None,
    ) -> dict[str, Any]:
        auth = (auth_header or "").strip()
        if not auth.lower().startswith("bearer "):
            raise web.HTTPUnauthorized(
                text=json.dumps(
                    {"error": "missing Authorization: Bearer <userId>::<apiKey>"}
                ),
                content_type="application/json",
            )

        token = auth.split(" ", 1)[1].strip()
        user_id, api_key = self._parse_compound_key(token)
        if not user_id or not api_key:
            raise web.HTTPUnauthorized(
                text=json.dumps(
                    {"error": "invalid bearer token format; expected userId::apiKey"}
                ),
                content_type="application/json",
            )

        headers = {"x-api-key": token}
        email = (user_email or "").strip().lower()
        if email:
            headers["x-user-email"] = email

        try:
            timeout = ClientTimeout(total=DB_API_TIMEOUT_SECONDS)
            async with ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._db_url("/backend-api/ping"),
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        raise web.HTTPUnauthorized(
                            text=json.dumps({"error": "invalid api credentials"}),
                            content_type="application/json",
                        )
                    try:
                        data = await response.json()
                    except Exception as exc:
                        raise web.HTTPBadGateway(
                            text=json.dumps(
                                {"error": "auth verification invalid response"}
                            ),
                            content_type="application/json",
                        ) from exc
        except web.HTTPException:
            raise
        except Exception as exc:
            raise web.HTTPBadGateway(
                text=json.dumps({"error": "auth verification failed"}),
                content_type="application/json",
            ) from exc

        if isinstance(data, dict) and "user_id" not in data:
            data["user_id"] = data.get("userId") or user_id
        return data

    async def _verify_web_key(self, web_key: str) -> dict[str, Any]:
        web_key = (web_key or "").strip()
        if not web_key:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "missing x-web-key"}),
                content_type="application/json",
            )

        user_id, secret = self._parse_compound_key(web_key)
        if not user_id or not secret:
            raise web.HTTPUnauthorized(
                text=json.dumps(
                    {"error": "invalid x-web-key format; expected userId::webKey"}
                ),
                content_type="application/json",
            )

        try:
            timeout = ClientTimeout(total=DB_API_TIMEOUT_SECONDS)
            async with ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._db_url("/web-api/whoami"),
                    headers={"x-web-key": web_key},
                ) as response:
                    if response.status != 200:
                        raise web.HTTPUnauthorized(
                            text=json.dumps({"error": "invalid web key"}),
                            content_type="application/json",
                        )
                    try:
                        data = await response.json()
                    except Exception as exc:
                        raise web.HTTPBadGateway(
                            text=json.dumps(
                                {"error": "auth verification invalid response"}
                            ),
                            content_type="application/json",
                        ) from exc
        except web.HTTPException:
            raise
        except Exception as exc:
            raise web.HTTPBadGateway(
                text=json.dumps({"error": "auth verification failed"}),
                content_type="application/json",
            ) from exc

        if isinstance(data, dict) and "user_id" not in data:
            data["user_id"] = user_id
        return data

    async def require_user(self, request: web.Request) -> dict[str, Any]:
        if not REQUIRE_AUTH:
            return {"auth_bypassed": True}

        web_key = (request.headers.get("x-web-key") or "").strip()
        if web_key:
            return await self._verify_web_key(web_key)

        return await self._verify_api_bearer(
            request.headers.get("Authorization", ""),
            request.headers.get("x-user-email"),
        )

    def load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        if self.model_loaded and self.model is not None:
            return

        is_local_file = (
            self.model_name_or_path.endswith(".nemo")
            or os.path.exists(self.model_name_or_path)
        )

        if is_local_file:
            logger.info(f"Loading model from local file: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                self.model_name_or_path,
                map_location="cpu",
            )
        else:
            logger.info(f"Loading model from HuggingFace: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name_or_path,
                map_location="cpu",
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This service expects a GPU-enabled runtime."
            )

        self.model = self.model.cuda()

        logger.info(
            "Setting att_context_size=[70, {}] ({})",
            self.right_context,
            RIGHT_CONTEXT_OPTIONS.get(self.right_context, "custom"),
        )
        self.model.encoder.set_default_att_context_size([70, self.right_context])

        logger.info("Configuring greedy decoding for Blackwell compatibility...")
        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create(
                {
                    "strategy": "greedy",
                    "greedy": {
                        "max_symbols": 10,
                        "loop_labels": False,
                        "use_cuda_graph_decoder": False,
                    },
                }
            )
        )
        self.model.eval()
        self.model.preprocessor.featurizer.dither = 0.0

        scfg = self.model.encoder.streaming_cfg
        logger.info(
            "Streaming config: chunk_size={}, shift_size={}",
            scfg.chunk_size,
            scfg.shift_size,
        )

        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get("window_stride", 0.01)
        self.hop_samples = int(hop_length_sec * self.sample_rate)

        self.shift_frames = (
            scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
        )
        pre_cache = scfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
        self.drop_extra = scfg.drop_extra_pre_encoded

        self.final_padding_frames = (self.right_context + 1) * self.shift_frames
        padding_ms = self.final_padding_frames * hop_length_sec * 1000

        self.overlap_samples = self.pre_encode_cache_size * self.hop_samples
        overlap_ms = self.overlap_samples * 1000 / self.sample_rate

        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info("Model loaded: {}", type(self.model).__name__)
        logger.info("Shift size: {:.0f}ms ({} frames)", shift_ms, self.shift_frames)
        logger.info("Pre-encode cache: {} frames", self.pre_encode_cache_size)
        logger.info(
            "Final chunk padding: {:.0f}ms ({} frames)",
            padding_ms,
            self.final_padding_frames,
        )
        logger.info(
            "Audio overlap for resets: {:.0f}ms ({} samples)",
            overlap_ms,
            self.overlap_samples,
        )

        self._warmup()
        self.model_loaded = True
        self._mark_activity()

    def load_vad_model(self):
        """Load the NeMo VAD model used for batch voice activity detection."""
        if self.vad_model is not None:
            return

        logger.info("Loading VAD model from pretrained checkpoint: {}", self.vad_model_name)
        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            model_name=self.vad_model_name,
            map_location="cpu",
        )
        vad_model = vad_model.eval()
        self.vad_model = vad_model

        labels = list(getattr(vad_model.cfg, "labels", []) or [])
        if "speech" in labels:
            self.vad_speech_label_index = labels.index("speech")
        else:
            self.vad_speech_label_index = 1

        logger.info(
            "VAD model loaded: {} (speech_label_index={})",
            type(vad_model).__name__,
            self.vad_speech_label_index,
        )
        self._mark_activity()

    def load_diarization_model(self):
        """Load the NeMo Sortformer diarization model used for speaker detection."""
        if self.diarization_model is not None:
            return

        logger.info(
            "Loading diarization model from pretrained checkpoint: {}",
            self.diarization_model_name,
        )
        diarization_model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(
            model_name=self.diarization_model_name,
            map_location="cpu",
        )
        diarization_model = diarization_model.eval()
        self.diarization_model = diarization_model
        logger.info("Diarization model loaded: {}", type(diarization_model).__name__)
        self._mark_activity()

    def unload_model(self):
        """Release GPU/model resources until the next request cold starts them."""
        if self.model is None:
            self.model_loaded = False
            return

        logger.info(
            "Unloading ASR model after {:.0f}s of inactivity",
            time.monotonic() - self.last_activity_at,
        )

        model = self.model
        self.model = None
        self.model_loaded = False
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_vad_model(self):
        """Release VAD model resources until the next VAD request loads them again."""
        if self.vad_model is None:
            return

        logger.info("Unloading VAD model after {:.0f}s of inactivity", time.monotonic() - self.last_activity_at)
        vad_model = self.vad_model
        self.vad_model = None
        del vad_model
        gc.collect()

    def unload_diarization_model(self):
        """Release the diarization model until the next request loads it again."""
        if self.diarization_model is None:
            return

        logger.info(
            "Unloading diarization model after {:.0f}s of inactivity",
            time.monotonic() - self.last_activity_at,
        )
        diarization_model = self.diarization_model
        self.diarization_model = None
        del diarization_model
        gc.collect()

    async def ensure_model_loaded(self):
        """Cold start the model only when a real request needs it."""
        if self.model_loaded and self.model is not None:
            self._mark_activity()
            return

        async with self.inference_lock:
            if self.model_loaded and self.model is not None:
                self._mark_activity()
                return

            logger.info("Cold starting ASR model on demand...")
            await asyncio.get_event_loop().run_in_executor(None, self.load_model)
            self._mark_activity()

    async def ensure_vad_model_loaded(self):
        """Cold start the VAD model only when a request needs it."""
        if self.vad_model is not None:
            self._mark_activity()
            return

        async with self.inference_lock:
            if self.vad_model is not None:
                self._mark_activity()
                return

            logger.info("Cold starting VAD model on demand...")
            await asyncio.get_event_loop().run_in_executor(None, self.load_vad_model)
            self._mark_activity()

    async def ensure_diarization_model_loaded(self):
        """Cold start the diarization model only when a request needs it."""
        if self.diarization_model is not None:
            self._mark_activity()
            return

        async with self.inference_lock:
            if self.diarization_model is not None:
                self._mark_activity()
                return

            logger.info("Cold starting diarization model on demand...")
            await asyncio.get_event_loop().run_in_executor(None, self.load_diarization_model)
            self._mark_activity()

    async def idle_monitor(self):
        """Unload the model after a quiet period with no active work."""
        logger.info(
            "Idle monitor active: unload after {}s of inactivity",
            self.idle_unload_seconds,
        )
        while True:
            await asyncio.sleep(self.idle_check_interval_seconds)
            if self.idle_unload_seconds <= 0:
                continue
            if (
                (not self.model_loaded or self.model is None)
                and self.vad_model is None
                and self.diarization_model is None
            ):
                continue
            if self._is_busy():
                continue

            idle_for = time.monotonic() - self.last_activity_at
            if idle_for < self.idle_unload_seconds:
                continue

            async with self.inference_lock:
                idle_for = time.monotonic() - self.last_activity_at
                if not self._is_busy() and idle_for >= self.idle_unload_seconds:
                    if self.model_loaded and self.model is not None:
                        await asyncio.get_event_loop().run_in_executor(None, self.unload_model)
                    if self.vad_model is not None:
                        await asyncio.get_event_loop().run_in_executor(None, self.unload_vad_model)
                    if self.diarization_model is not None:
                        await asyncio.get_event_loop().run_in_executor(None, self.unload_diarization_model)

    def _warmup(self):
        """Run warmup inference using the streaming API to claim GPU memory."""
        logger.info("Running warmup inference (streaming API) to claim GPU memory...")
        start = time.perf_counter()

        warmup_samples = self.sample_rate + (self.final_padding_frames * self.hop_samples)
        warmup_audio = np.zeros(warmup_samples, dtype=np.float32)

        with torch.inference_mode():
            audio_tensor = torch.from_numpy(warmup_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(warmup_audio)], device="cuda")
            mel, mel_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
            cache = self.model.encoder.get_initial_cache_state(batch_size=1)

            _ = self.model.conformer_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=True,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Warmup complete in {:.0f}ms - GPU memory claimed", elapsed)

    def _init_session(self, session: ASRSession):
        """Initialize a fresh session."""
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]

        if session.overlap_buffer is not None and len(session.overlap_buffer) > 0:
            session.accumulated_audio = session.overlap_buffer.copy()
            overlap_ms = len(session.overlap_buffer) * 1000 / self.sample_rate
            logger.debug(
                "Session {}: prepending {} samples ({:.0f}ms) of overlap audio",
                session.id,
                len(session.overlap_buffer),
                overlap_ms,
            )
            session.overlap_buffer = None
        else:
            session.accumulated_audio = np.array([], dtype=np.float32)

        session.accumulated_audio_base_samples = session.total_samples_received
        session.emitted_frames = 0
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""
        if session.diarization_audio is None:
            session.diarization_audio = np.array([], dtype=np.float32)
        self._reset_vad_state(session)

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_float(value: Any, default: float) -> float:
        if value is None or value == "":
            return default
        return float(value)

    def _mode_uses_asr(self, mode: str) -> bool:
        return mode in {"asr", "asr_vad"}

    def _mode_uses_vad(self, mode: str) -> bool:
        return mode in {"vad", "asr_vad"}

    @staticmethod
    def _speaker_label(value: Any) -> str:
        if isinstance(value, str):
            return value if value.startswith("speaker_") else f"speaker_{value}"
        if isinstance(value, (int, np.integer)):
            return f"speaker_{int(value)}"
        try:
            numeric = int(value)
            return f"speaker_{numeric}"
        except Exception:
            text = str(value).strip()
            return text if text.startswith("speaker_") else f"speaker_{text or 'unknown'}"

    def _session_uses_diarization(self, session: ASRSession) -> bool:
        return bool(session.diarization_enabled)

    def _reset_vad_state(self, session: ASRSession):
        session.vad_processed_samples = 0
        session.vad_speech_active = False
        session.vad_active_start_ms = None
        session.vad_candidate_start_ms = None
        session.vad_candidate_last_speech_end_ms = None

    def _configure_session(self, session: ASRSession, data: dict[str, Any]):
        mode = str(data.get("mode", session.websocket_mode or "asr")).strip().lower()
        if mode not in {"asr", "vad", "asr_vad"}:
            raise ValueError("mode must be one of: asr, vad, asr_vad")

        vad_cfg = data.get("vad") or {}
        session.websocket_mode = mode
        session.emit_vad_frames = self._parse_bool(
            data.get("emit_vad_frames", session.emit_vad_frames),
            default=session.emit_vad_frames,
        )
        session.auto_finalize_on_vad_end = self._parse_bool(
            data.get("auto_finalize_on_vad_end", session.auto_finalize_on_vad_end),
            default=(mode == "asr_vad"),
        )
        session.diarization_enabled = self._parse_bool(
            data.get("enable_diarization", session.diarization_enabled),
            default=session.diarization_enabled,
        )
        session.emit_diarization_segments = self._parse_bool(
            data.get("emit_diarization_segments", session.emit_diarization_segments),
            default=session.emit_diarization_segments,
        )
        session.vad_threshold = self._parse_float(
            vad_cfg.get("threshold", data.get("vad_threshold")),
            session.vad_threshold,
        )
        session.vad_window_seconds = self._parse_float(
            vad_cfg.get("window_seconds", data.get("vad_window_seconds")),
            session.vad_window_seconds,
        )
        session.vad_hop_seconds = self._parse_float(
            vad_cfg.get("hop_seconds", data.get("vad_hop_seconds")),
            session.vad_hop_seconds,
        )
        session.vad_min_speech_seconds = self._parse_float(
            vad_cfg.get("min_speech_seconds", data.get("vad_min_speech_seconds")),
            session.vad_min_speech_seconds,
        )
        session.vad_min_silence_seconds = self._parse_float(
            vad_cfg.get("min_silence_seconds", data.get("vad_min_silence_seconds")),
            session.vad_min_silence_seconds,
        )

        if session.vad_threshold < 0.0 or session.vad_threshold > 1.0:
            raise ValueError("vad threshold must be between 0 and 1")
        if session.vad_window_seconds <= 0 or session.vad_hop_seconds <= 0:
            raise ValueError("vad window_seconds and hop_seconds must be positive")
        if session.vad_min_speech_seconds < 0 or session.vad_min_silence_seconds < 0:
            raise ValueError("vad min_speech_seconds and min_silence_seconds must be non-negative")

    def _build_session_updated_payload(self, session: ASRSession) -> dict[str, Any]:
        return {
            "message_type": "session_updated",
            "session_id": session.id,
            "mode": session.websocket_mode,
            "vad": {
                "enabled": self._mode_uses_vad(session.websocket_mode),
                "emit_vad_frames": session.emit_vad_frames,
                "auto_finalize_on_vad_end": session.auto_finalize_on_vad_end,
                "threshold": session.vad_threshold,
                "window_seconds": session.vad_window_seconds,
                "hop_seconds": session.vad_hop_seconds,
                "min_speech_seconds": session.vad_min_speech_seconds,
                "min_silence_seconds": session.vad_min_silence_seconds,
                "model": self.vad_model_name,
            },
            "diarization": {
                "enabled": session.diarization_enabled,
                "emit_diarization_segments": session.emit_diarization_segments,
                "model": self.diarization_model_name,
            },
        }

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket client connection."""
        await self.require_user(request)

        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        session_id = str(uuid.uuid4())[:8]
        session = ASRSession(id=session_id, websocket=ws)
        self._configure_session(
            session,
            {
                "mode": request.query.get("mode", "asr"),
                "emit_vad_frames": request.query.get("emit_vad_frames"),
                "auto_finalize_on_vad_end": request.query.get("auto_finalize_on_vad_end"),
                "enable_diarization": request.query.get("enable_diarization"),
                "emit_diarization_segments": request.query.get("emit_diarization_segments"),
                "vad_threshold": request.query.get("vad_threshold"),
                "vad_window_seconds": request.query.get("vad_window_seconds"),
                "vad_hop_seconds": request.query.get("vad_hop_seconds"),
                "vad_min_speech_seconds": request.query.get("vad_min_speech_seconds"),
                "vad_min_silence_seconds": request.query.get("vad_min_silence_seconds"),
            },
        )
        self.sessions[session_id] = session
        logger.info("Client {} connected", session_id)

        try:
            if self._mode_uses_asr(session.websocket_mode):
                await self.ensure_model_loaded()
                async with self.inference_lock:
                    await asyncio.get_event_loop().run_in_executor(None, self._init_session, session)
            else:
                session.accumulated_audio = np.array([], dtype=np.float32)
                session.accumulated_audio_base_samples = session.total_samples_received
                session.current_text = ""
                self._reset_vad_state(session)

            if self._mode_uses_vad(session.websocket_mode):
                await self.ensure_vad_model_loaded()
            if self._session_uses_diarization(session):
                await self.ensure_diarization_model_loaded()
                session.diarization_audio = np.array([], dtype=np.float32)
                session.diarization_last_emitted_end_ms = 0

            await ws.send_str(json.dumps(self._build_session_updated_payload(session)))

            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        message_type = data.get("message_type")

                        if message_type == "input_audio_chunk" and data.get("audio_base_64"):
                            audio_bytes = base64.b64decode(data["audio_base_64"])
                            await self._handle_audio(session, audio_bytes)
                        elif msg_type == "configure":
                            self._configure_session(session, data)
                            if self._mode_uses_asr(session.websocket_mode):
                                await self.ensure_model_loaded()
                                async with self.inference_lock:
                                    await asyncio.get_event_loop().run_in_executor(None, self._init_session, session)
                            else:
                                session.accumulated_audio = np.array([], dtype=np.float32)
                                session.accumulated_audio_base_samples = session.total_samples_received
                                session.current_text = ""
                                self._reset_vad_state(session)
                            if self._mode_uses_vad(session.websocket_mode):
                                await self.ensure_vad_model_loaded()
                            if self._session_uses_diarization(session):
                                await self.ensure_diarization_model_loaded()
                                if session.diarization_audio is None:
                                    session.diarization_audio = np.array([], dtype=np.float32)
                            else:
                                session.diarization_audio = None
                                session.diarization_last_emitted_end_ms = 0
                            await ws.send_str(json.dumps(self._build_session_updated_payload(session)))
                        elif msg_type == "reset" or msg_type == "end":
                            finalize = data.get("finalize", True)
                            await self._reset_session(session, finalize=finalize)
                        else:
                            logger.warning(
                                "Client {}: unknown message type: {}",
                                session_id,
                                msg_type or message_type,
                            )
                    except json.JSONDecodeError:
                        logger.warning("Client {}: invalid JSON", session_id)
                elif msg.type == WSMsgType.ERROR:
                    logger.error("Client {} WebSocket error: {}", session_id, ws.exception())
                    break

            logger.info("Client {} disconnected", session_id)
        except Exception as exc:
            logger.error("Client {} error: {}", session_id, exc)
            logger.error(traceback.format_exc())
            try:
                await ws.send_str(json.dumps({"message_type": "error", "error": str(exc)}))
            except Exception:
                pass
        finally:
            if session.websocket is not None and self._mode_uses_vad(session.websocket_mode):
                for event in self._flush_vad_state(session):
                    with contextlib.suppress(Exception):
                        await ws.send_str(json.dumps(event))
            self.sessions.pop(session_id, None)
            self._mark_activity()

        return ws

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames are available."""
        self._mark_activity()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(
                "Session {}: recv chunk {}B hash={}",
                session.id,
                len(audio_bytes),
                chunk_hash,
            )

        session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_np])
        session.total_samples_received += len(audio_np)
        if self._session_uses_diarization(session):
            if session.diarization_audio is None:
                session.diarization_audio = np.array([], dtype=np.float32)
            session.diarization_audio = np.concatenate([session.diarization_audio, audio_np])

        if self._mode_uses_vad(session.websocket_mode):
            async with self.inference_lock:
                vad_events = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._process_vad_windows,
                    session,
                )
            should_auto_finalize = False
            speaker_range_ms: Optional[tuple[int, int]] = None
            for event in vad_events:
                await session.websocket.send_str(json.dumps(event))
                if (
                    event.get("message_type") == "speech_ended"
                    and self._mode_uses_asr(session.websocket_mode)
                    and session.auto_finalize_on_vad_end
                ):
                    should_auto_finalize = True
                    speaker_range_ms = (
                        int(event.get("start_ms", 0)),
                        int(event.get("end_ms", 0)),
                    )

            if should_auto_finalize:
                await self._reset_session(session, finalize=True, speaker_range_ms=speaker_range_ms)
                return

        if not self._mode_uses_asr(session.websocket_mode):
            return

        min_audio_for_chunk = (
            session.emitted_frames + self.shift_frames + 1
        ) * self.hop_samples

        while len(session.accumulated_audio) >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._process_chunk,
                    session,
                )

            if text is not None and text != session.current_text:
                session.current_text = text
                logger.debug(
                    "Session {} interim: {}",
                    session.id,
                    text[-50:] if len(text) > 50 else text,
                )
                await session.websocket.send_str(
                    json.dumps({"message_type": "partial_transcript", "text": text})
                )

            min_audio_for_chunk = (
                session.emitted_frames + self.shift_frames + 1
            ) * self.hop_samples

    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process accumulated audio and run streaming inference."""
        try:
            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device="cuda")

            if DEBUG_ASR:
                audio_hash = _hash_audio(session.accumulated_audio)
                logger.debug(
                    "Session {}: process audio={} hash={}",
                    session.id,
                    len(session.accumulated_audio),
                    audio_hash,
                )

            with torch.inference_mode():
                mel, _ = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

                if DEBUG_ASR:
                    mel_hash = hashlib.md5(mel.cpu().numpy().tobytes()).hexdigest()[:8]
                    logger.debug(
                        "Session {}: mel shape={} hash={}",
                        session.id,
                        mel.shape[-1],
                        mel_hash,
                    )

                available_frames = mel.shape[-1] - 1
                new_frame_count = available_frames - session.emitted_frames
                if new_frame_count < self.shift_frames:
                    return session.current_text

                if session.emitted_frames == 0:
                    chunk_start = 0
                    chunk_end = self.shift_frames
                    drop_extra = 0
                else:
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    chunk_end = session.emitted_frames + self.shift_frames
                    drop_extra = self.drop_extra

                chunk_mel = mel[:, :, chunk_start:chunk_end]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device="cuda")

                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

                session.emitted_frames += self.shift_frames

                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, "text"):
                        return hyp.text
                    if isinstance(hyp, str):
                        return hyp
                    return str(hyp)

                return session.current_text
        except Exception as exc:
            logger.error("Session {} chunk processing error: {}", session.id, exc)
            logger.error(traceback.format_exc())
            return None

    async def _reset_session(
        self,
        session: ASRSession,
        finalize: bool = True,
        speaker_range_ms: Optional[tuple[int, int]] = None,
    ):
        """Handle reset with soft or hard finalization."""
        audio_samples = len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            "Session {} {} reset: accumulated={} samples ({}ms), emitted={} frames",
            session.id,
            "hard" if finalize else "soft",
            audio_samples,
            audio_duration_ms,
            session.emitted_frames,
        )

        if not finalize:
            text = session.current_text
            if self._mode_uses_asr(session.websocket_mode):
                payload: dict[str, Any] = {"message_type": "committed_transcript", "text": text}
                if self._session_uses_diarization(session) and session.diarization_audio is not None and len(session.diarization_audio) > 0:
                    async with self.inference_lock:
                        new_segments, best_segment, diarization_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            self._compute_diarization_updates_for_session,
                            session,
                            speaker_range_ms,
                        )
                    if session.emit_diarization_segments:
                        for segment in new_segments:
                            await session.websocket.send_str(
                                json.dumps({"message_type": "speaker_segment", **segment})
                            )
                    payload["diarization"] = diarization_result
                    if best_segment is not None:
                        payload["speaker"] = best_segment["speaker"]
                        payload["speaker_segment"] = best_segment
                await session.websocket.send_str(json.dumps(payload))
            for event in self._flush_vad_state(session):
                await session.websocket.send_str(json.dumps(event))
            logger.debug(
                "Session {} soft reset: '{}'",
                session.id,
                text[-50:] if len(text) > 50 else text,
            )
            return

        original_audio_length = (
            len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        )

        if original_audio_length > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate(
                [session.accumulated_audio, silence_padding]
            )

        final_text = session.current_text
        if session.accumulated_audio is not None and len(session.accumulated_audio) > 0:
            start_time = time.perf_counter()
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._process_final_chunk,
                    session,
                )
                if text is not None:
                    final_text = text
                    session.current_text = text
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "Session {} final chunk processed in {:.1f}ms: '{}'",
                session.id,
                elapsed_ms,
                final_text[-50:] if len(final_text) > 50 else final_text,
            )

        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text) :].lstrip()
        else:
            delta_text = final_text
            logger.debug(
                "Session {}: ASR correction detected, last='{}', new='{}'",
                session.id,
                session.last_emitted_text[-30:],
                final_text[-30:],
            )

        diarization_result = None
        best_speaker_segment = None
        if self._session_uses_diarization(session) and session.diarization_audio is not None and len(session.diarization_audio) > 0:
            async with self.inference_lock:
                new_segments, best_speaker_segment, diarization_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._compute_diarization_updates_for_session,
                    session,
                    speaker_range_ms,
                )
            if session.emit_diarization_segments:
                for segment in new_segments:
                    await session.websocket.send_str(
                        json.dumps({"message_type": "speaker_segment", **segment})
                    )

        session.last_emitted_text = final_text
        committed_payload: dict[str, Any] = {
            "message_type": "committed_transcript",
            "text": delta_text,
        }
        if diarization_result is not None:
            committed_payload["diarization"] = diarization_result
        if best_speaker_segment is not None:
            committed_payload["speaker"] = best_speaker_segment["speaker"]
            committed_payload["speaker_segment"] = best_speaker_segment
        await session.websocket.send_str(json.dumps(committed_payload))

        logger.debug(
            "Session {} hard reset: delta='{}' (cumulative='{}')",
            session.id,
            delta_text,
            final_text[-50:] if len(final_text) > 50 else final_text,
        )

        for event in self._flush_vad_state(session):
            await session.websocket.send_str(json.dumps(event))

        session.last_emitted_text = ""
        session.overlap_buffer = None
        if self._mode_uses_asr(session.websocket_mode):
            self._init_session(session)
        else:
            session.accumulated_audio = np.array([], dtype=np.float32)
            session.current_text = ""
            self._reset_vad_state(session)
        logger.debug("Session {} hard reset complete, state fully reset for next turn", session.id)

    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process all remaining audio with keep_all_outputs=True."""
        try:
            if len(session.accumulated_audio) == 0:
                return session.current_text

            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device="cuda")

            with torch.inference_mode():
                mel, _ = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
                total_mel_frames = mel.shape[-1]
                remaining_frames = total_mel_frames - session.emitted_frames

                logger.debug(
                    "Session {} final chunk: total_mel={}, emitted={}, remaining={}",
                    session.id,
                    total_mel_frames,
                    session.emitted_frames,
                    remaining_frames,
                )

                if remaining_frames <= 0:
                    logger.warning("Session {}: No remaining frames to process!", session.id)
                    return session.current_text

                if session.emitted_frames == 0:
                    chunk_start = 0
                    drop_extra = 0
                else:
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    drop_extra = self.drop_extra

                chunk_mel = mel[:, :, chunk_start:]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device="cuda")

                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=True,
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, "text"):
                        final_text = hyp.text
                    elif isinstance(hyp, str):
                        final_text = hyp
                    else:
                        final_text = str(hyp)

                    logger.debug(
                        "Session {} final chunk output: '{}' (was: '{}')",
                        session.id,
                        final_text[-50:] if len(final_text) > 50 else final_text,
                        session.current_text[-30:]
                        if len(session.current_text) > 30
                        else session.current_text,
                    )
                    return final_text

                logger.debug("Session {} final chunk: no new text from model", session.id)
                return session.current_text
        except Exception as exc:
            logger.error("Session {} final chunk error: {}", session.id, exc)
            logger.error(traceback.format_exc())
            return None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy" if (self.model_loaded or self.vad_model is not None or self.diarization_model is not None) else "idle",
                "model_loaded": self.model_loaded,
                "model": self.model_name_or_path,
                "vad_model_loaded": self.vad_model is not None,
                "vad_model": self.vad_model_name,
                "diarization_model_loaded": self.diarization_model is not None,
                "diarization_model": self.diarization_model_name,
                "right_context": self.right_context,
                "sample_rate": self.sample_rate,
                "busy": self._is_busy(),
                "idle_unload_seconds": self.idle_unload_seconds,
                "preload_model": self.preload_model,
            }
        )

    async def root_handler(self, request: web.Request) -> web.Response:
        """Basic endpoint inventory."""
        return web.json_response(
            {
                "service": "nemo-asr",
                "status": "ok",
                "endpoints": {
                    "health": "/health",
                    "websocket": "/ws",
                    "api": "/api",
                    "transcribe": "/api/transcribe",
                    "vad": "/api/vad",
                    "diarize": "/api/diarize",
                },
            }
        )

    async def api_info_handler(self, request: web.Request) -> web.Response:
        """Return API information and example payloads."""
        await self.require_user(request)
        return web.json_response(self._build_api_description())

    def _convert_media_to_audio_array(self, media_path: Path) -> np.ndarray:
        """Use ffmpeg to convert media into mono 16kHz PCM samples."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)

        try:
            command = [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(media_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                str(wav_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed to process media: {completed.stderr.strip() or completed.stdout.strip()}"
                )

            audio, sample_rate = sf.read(str(wav_path), dtype="float32")
            if sample_rate != self.sample_rate:
                raise RuntimeError(
                    f"Unexpected sample rate after conversion: {sample_rate}"
                )
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return np.asarray(audio, dtype=np.float32)
        finally:
            if wav_path.exists():
                wav_path.unlink()

    async def _download_remote_media(self, audio_url: str) -> Path:
        """Download a remote audio/video file to a temporary path."""
        suffix = Path(audio_url.split("?", 1)[0]).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as media_file:
            media_path = Path(media_file.name)

        timeout = ClientTimeout(total=300)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.get(audio_url) as response:
                    response.raise_for_status()
                    with media_path.open("wb") as handle:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            handle.write(chunk)
            return media_path
        except Exception:
            if media_path.exists():
                media_path.unlink()
            raise

    async def _save_uploaded_media(self, request: web.Request) -> tuple[Path, dict[str, Any]]:
        """Persist a multipart-uploaded audio or video file to a temporary path."""
        form = await request.post()
        file_field = None
        for key in ("file", "audio", "media", "upload", "source"):
            candidate = form.get(key)
            if candidate is not None and hasattr(candidate, "file") and hasattr(candidate, "filename"):
                file_field = candidate
                break

        if file_field is None:
            for _, candidate in form.items():
                if hasattr(candidate, "file") and hasattr(candidate, "filename"):
                    file_field = candidate
                    break

        if file_field is None:
            raise web.HTTPBadRequest(
                text=json.dumps(
                    {
                        "error": "missing_upload_file",
                        "message": "Provide a multipart file field such as file, audio, media, upload, or source",
                    }
                ),
                content_type="application/json",
            )

        filename = str(getattr(file_field, "filename", "") or "upload.bin").strip() or "upload.bin"
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as media_file:
            media_path = Path(media_file.name)

        file_obj = getattr(file_field, "file", None)
        if file_obj is None:
            media_path.unlink(missing_ok=True)
            raise web.HTTPBadRequest(
                text=json.dumps(
                    {"error": "invalid_upload_file", "message": "Uploaded file payload is not readable"}
                ),
                content_type="application/json",
            )

        try:
            file_obj.seek(0)
        except Exception:
            pass

        with media_path.open("wb") as handle:
            shutil.copyfileobj(file_obj, handle)

        meta: dict[str, Any] = {
            "input_mode": "upload",
            "source_filename": filename,
            "source_type": str(getattr(file_field, "content_type", "") or "").strip() or None,
            "source_size": media_path.stat().st_size if media_path.exists() else None,
            "mode": str(form.get("mode", "")).strip().lower() or None,
            "enable_diarization": str(form.get("enable_diarization", "")).strip() or None,
            "diarization_speakers": str(form.get("diarization_speakers", "")).strip() or None,
        }
        vad_payload = {}
        for key in (
            "threshold",
            "window_seconds",
            "hop_seconds",
            "min_speech_seconds",
            "min_silence_seconds",
        ):
            value = form.get(f"vad_{key}", form.get(key))
            if value not in (None, ""):
                vad_payload[key] = value
        if vad_payload:
            meta["vad"] = vad_payload

        return media_path, meta

    def _transcribe_audio_array(self, audio: np.ndarray) -> str:
        """Run whole-file transcription through the streaming pipeline."""
        session = ASRSession(id=f"api-{uuid.uuid4().hex[:8]}", websocket=None)
        self._init_session(session)
        session.accumulated_audio = np.asarray(audio, dtype=np.float32)

        if len(session.accumulated_audio) == 0:
            return ""

        min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples
        while len(session.accumulated_audio) >= min_audio_for_chunk:
            text = self._process_chunk(session)
            if text is not None:
                session.current_text = text
            min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

        padding_samples = self.final_padding_frames * self.hop_samples
        if padding_samples > 0:
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate(
                [session.accumulated_audio, silence_padding]
            )

        final_text = self._process_final_chunk(session)
        return (final_text if final_text is not None else session.current_text).strip()

    def _compute_diarization_result(self, audio: np.ndarray) -> dict[str, Any]:
        """Run Sortformer diarization on a mono 16kHz waveform."""
        if self.diarization_model is None:
            raise RuntimeError("Diarization model is not loaded")

        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return {
                "speaker_segments": [],
                "speaker_count": 0,
                "diarization_model": self.diarization_model_name,
            }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = Path(wav_file.name)

        try:
            sf.write(str(wav_path), audio, self.sample_rate)
            diarization_output = self.diarization_model.diarize(
                audio=str(wav_path),
                batch_size=1,
                verbose=False,
            )
            raw_segments = diarization_output[0] if diarization_output else []
            speaker_segments: list[dict[str, Any]] = []
            speaker_labels: set[str] = set()
            for raw_segment in raw_segments:
                if isinstance(raw_segment, str):
                    parts = raw_segment.strip().split()
                    if len(parts) < 3:
                        continue
                    start_seconds, end_seconds, speaker_idx = parts[:3]
                else:
                    if len(raw_segment) < 3:
                        continue
                    start_seconds, end_seconds, speaker_idx = raw_segment[:3]
                speaker = self._speaker_label(speaker_idx)
                segment = {
                    "start_ms": int(round(float(start_seconds) * 1000)),
                    "end_ms": int(round(float(end_seconds) * 1000)),
                    "speaker": speaker,
                }
                if segment["end_ms"] <= segment["start_ms"]:
                    continue
                speaker_segments.append(segment)
                speaker_labels.add(speaker)

            return {
                "speaker_segments": speaker_segments,
                "speaker_count": len(speaker_labels),
                "diarization_model": self.diarization_model_name,
            }
        finally:
            if wav_path.exists():
                wav_path.unlink()

    def _compute_diarization_updates_for_session(
        self,
        session: ASRSession,
        focus_range_ms: Optional[tuple[int, int]] = None,
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]]:
        """Re-run diarization on cumulative session audio and return new segments plus the best speaker for a focus range."""
        audio = session.diarization_audio
        if audio is None:
            audio = np.array([], dtype=np.float32)
        diarization_result = self._compute_diarization_result(audio)
        all_segments = diarization_result["speaker_segments"]

        new_segments = [
            segment
            for segment in all_segments
            if segment["end_ms"] > session.diarization_last_emitted_end_ms
        ]
        if new_segments:
            session.diarization_last_emitted_end_ms = max(
                session.diarization_last_emitted_end_ms,
                max(segment["end_ms"] for segment in new_segments),
            )

        best_segment: Optional[dict[str, Any]] = None
        if focus_range_ms is not None:
            focus_start_ms, focus_end_ms = focus_range_ms
            best_overlap = 0
            for segment in all_segments:
                overlap_start = max(focus_start_ms, segment["start_ms"])
                overlap_end = min(focus_end_ms, segment["end_ms"])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_segment = segment

        return new_segments, best_segment, diarization_result

    def _vad_logits_for_segment(self, segment: np.ndarray) -> np.ndarray:
        """Run a single fixed-size segment through the VAD model and return logits."""
        if self.vad_model is None:
            raise RuntimeError("VAD model is not loaded")

        audio = np.asarray(segment, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("VAD segment must be mono")

        audio_batch = torch.tensor(audio[None, :], device=self.vad_model.device, dtype=torch.float32)
        audio_len = torch.tensor([audio.shape[0]], device=self.vad_model.device)
        with torch.inference_mode():
            logits = self.vad_model.forward(input_signal=audio_batch, input_signal_length=audio_len)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        logits_np = logits.detach().cpu().numpy()
        if logits_np.ndim == 2:
            return logits_np[0]
        raise RuntimeError(f"Unexpected VAD logits shape: {logits_np.shape}")

    def _compute_vad_result(
        self,
        audio: np.ndarray,
        threshold: float,
        window_seconds: float,
        hop_seconds: float,
        min_speech_seconds: float,
        min_silence_seconds: float,
    ) -> dict[str, Any]:
        """Run sliding-window VAD and return frame probabilities plus merged speech segments."""
        if len(audio) == 0:
            return {
                "speech_segments": [],
                "frame_predictions": [],
                "speech_duration_ms": 0,
                "threshold": threshold,
                "window_seconds": window_seconds,
                "hop_seconds": hop_seconds,
                "min_speech_seconds": min_speech_seconds,
                "min_silence_seconds": min_silence_seconds,
                "vad_model": self.vad_model_name,
            }

        window_samples = max(1, int(round(window_seconds * self.sample_rate)))
        hop_samples = max(1, int(round(hop_seconds * self.sample_rate)))
        min_speech_ms = int(round(min_speech_seconds * 1000))
        min_silence_ms = int(round(min_silence_seconds * 1000))

        frame_predictions: list[dict[str, Any]] = []
        speech_windows: list[dict[str, Any]] = []

        for start_sample in range(0, len(audio), hop_samples):
            end_sample = min(len(audio), start_sample + window_samples)
            segment = audio[start_sample:end_sample]
            if len(segment) < window_samples:
                segment = np.pad(segment, (0, window_samples - len(segment)))

            logits = self._vad_logits_for_segment(segment)
            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            speech_prob = float(probs[self.vad_speech_label_index])

            frame_start_ms = int(round(start_sample * 1000 / self.sample_rate))
            frame_end_ms = int(round(end_sample * 1000 / self.sample_rate))
            is_speech = speech_prob >= threshold

            frame_entry = {
                "start_ms": frame_start_ms,
                "end_ms": frame_end_ms,
                "speech_probability": speech_prob,
                "is_speech": is_speech,
            }
            frame_predictions.append(frame_entry)
            if is_speech:
                speech_windows.append(frame_entry)

        merged_segments: list[dict[str, Any]] = []
        current_segment: Optional[dict[str, Any]] = None

        for frame in speech_windows:
            if current_segment is None:
                current_segment = {
                    "start_ms": frame["start_ms"],
                    "end_ms": frame["end_ms"],
                    "max_speech_probability": frame["speech_probability"],
                }
                continue

            gap_ms = frame["start_ms"] - current_segment["end_ms"]
            if gap_ms <= min_silence_ms:
                current_segment["end_ms"] = max(current_segment["end_ms"], frame["end_ms"])
                current_segment["max_speech_probability"] = max(
                    current_segment["max_speech_probability"],
                    frame["speech_probability"],
                )
            else:
                if current_segment["end_ms"] - current_segment["start_ms"] >= min_speech_ms:
                    merged_segments.append(current_segment)
                current_segment = {
                    "start_ms": frame["start_ms"],
                    "end_ms": frame["end_ms"],
                    "max_speech_probability": frame["speech_probability"],
                }

        if current_segment is not None and current_segment["end_ms"] - current_segment["start_ms"] >= min_speech_ms:
            merged_segments.append(current_segment)

        speech_duration_ms = sum(segment["end_ms"] - segment["start_ms"] for segment in merged_segments)
        return {
            "speech_segments": merged_segments,
            "frame_predictions": frame_predictions,
            "speech_duration_ms": speech_duration_ms,
            "threshold": threshold,
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "min_speech_seconds": min_speech_seconds,
            "min_silence_seconds": min_silence_seconds,
            "vad_model": self.vad_model_name,
        }

    def _process_vad_windows(self, session: ASRSession) -> list[dict[str, Any]]:
        """Process newly accumulated audio through VAD and emit websocket events."""
        if self.vad_model is None or not self._mode_uses_vad(session.websocket_mode):
            return []

        events: list[dict[str, Any]] = []
        window_samples = max(1, int(round(session.vad_window_seconds * self.sample_rate)))
        hop_samples = max(1, int(round(session.vad_hop_seconds * self.sample_rate)))
        min_speech_ms = int(round(session.vad_min_speech_seconds * 1000))
        min_silence_ms = int(round(session.vad_min_silence_seconds * 1000))

        while session.vad_processed_samples + window_samples <= len(session.accumulated_audio):
            start_sample = session.vad_processed_samples
            end_sample = start_sample + window_samples
            segment = session.accumulated_audio[start_sample:end_sample]
            logits = self._vad_logits_for_segment(segment)
            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            speech_prob = float(probs[self.vad_speech_label_index])
            is_speech = speech_prob >= session.vad_threshold

            absolute_start_sample = session.accumulated_audio_base_samples + start_sample
            absolute_end_sample = session.accumulated_audio_base_samples + end_sample
            frame_start_ms = int(round(absolute_start_sample * 1000 / self.sample_rate))
            frame_end_ms = int(round(absolute_end_sample * 1000 / self.sample_rate))

            if session.emit_vad_frames:
                events.append(
                    {
                        "message_type": "vad_frame",
                        "start_ms": frame_start_ms,
                        "end_ms": frame_end_ms,
                        "speech_probability": speech_prob,
                        "is_speech": is_speech,
                    }
                )

            if is_speech:
                if session.vad_candidate_start_ms is None:
                    session.vad_candidate_start_ms = frame_start_ms
                session.vad_candidate_last_speech_end_ms = frame_end_ms
                if (
                    not session.vad_speech_active
                    and session.vad_candidate_last_speech_end_ms is not None
                    and session.vad_candidate_last_speech_end_ms - session.vad_candidate_start_ms >= min_speech_ms
                ):
                    session.vad_speech_active = True
                    session.vad_active_start_ms = session.vad_candidate_start_ms
                    events.append(
                        {
                            "message_type": "speech_started",
                            "start_ms": session.vad_active_start_ms,
                            "speech_probability": speech_prob,
                        }
                    )
            elif session.vad_candidate_start_ms is not None and session.vad_candidate_last_speech_end_ms is not None:
                gap_ms = frame_start_ms - session.vad_candidate_last_speech_end_ms
                if gap_ms > min_silence_ms:
                    if session.vad_speech_active and session.vad_active_start_ms is not None:
                        end_ms = session.vad_candidate_last_speech_end_ms
                        events.append(
                            {
                                "message_type": "speech_ended",
                                "start_ms": session.vad_active_start_ms,
                                "end_ms": end_ms,
                                "duration_ms": end_ms - session.vad_active_start_ms,
                            }
                        )
                    session.vad_speech_active = False
                    session.vad_active_start_ms = None
                    session.vad_candidate_start_ms = None
                    session.vad_candidate_last_speech_end_ms = None

            session.vad_processed_samples += hop_samples

        return events

    def _flush_vad_state(self, session: ASRSession) -> list[dict[str, Any]]:
        """Emit any pending speech-ended event and clear VAD state."""
        events: list[dict[str, Any]] = []
        if session.vad_speech_active and session.vad_active_start_ms is not None:
            end_ms = (
                session.vad_candidate_last_speech_end_ms
                if session.vad_candidate_last_speech_end_ms is not None
                else int(
                    round(
                        (
                            session.accumulated_audio_base_samples
                            + (len(session.accumulated_audio) if session.accumulated_audio is not None else 0)
                        )
                        * 1000
                        / self.sample_rate
                    )
                )
            )
            events.append(
                {
                    "message_type": "speech_ended",
                    "start_ms": session.vad_active_start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - session.vad_active_start_ms,
                }
            )
        self._reset_vad_state(session)
        return events

    async def _api_inference_handler(
        self,
        request: web.Request,
        forced_mode: Optional[str] = None,
        force_diarization: bool = False,
    ) -> web.Response:
        """Download media from a URL and return ASR, VAD, and optional diarization results."""
        await self.require_user(request)

        payload: dict[str, Any] = {}
        media_path: Optional[Path] = None
        audio_url = ""
        input_mode = "url"
        source_filename: Optional[str] = None
        source_type: Optional[str] = None
        source_size: Optional[int] = None

        if request.content_type.startswith("multipart/"):
            try:
                media_path, payload = await self._save_uploaded_media(request)
            except web.HTTPException:
                raise
            input_mode = "upload"
            source_filename = payload.get("source_filename")
            source_type = payload.get("source_type")
            source_size = payload.get("source_size")
        else:
            try:
                payload = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                return web.json_response(
                    {
                        "error": "invalid_request_body",
                        "message": "Request body must be valid JSON or multipart form-data with an uploaded audio file",
                    },
                    status=400,
                )

            audio_url = str(payload.get("audio_url", "")).strip()
            if not audio_url or not audio_url.startswith(("http://", "https://")):
                return web.json_response(
                    {
                        "error": "invalid_audio_url",
                        "message": "Provide an http(s) audio_url pointing to an audio or video file, or upload a file as multipart form-data",
                    },
                    status=400,
                )

        mode = str(forced_mode or payload.get("mode", "asr")).strip().lower()
        if mode not in {"asr", "vad", "asr_vad", "diarize"}:
            return web.json_response(
                {
                    "error": "invalid_mode",
                    "message": "mode must be one of: asr, vad, asr_vad, diarize",
                },
                status=400,
            )

        enable_diarization = True if force_diarization else self._parse_bool(
            payload.get("enable_diarization"),
            default=False,
        )

        vad_payload = payload.get("vad") or {}
        threshold = float(vad_payload.get("threshold", DEFAULT_VAD_THRESHOLD))
        window_seconds = float(vad_payload.get("window_seconds", DEFAULT_VAD_WINDOW_SECONDS))
        hop_seconds = float(vad_payload.get("hop_seconds", DEFAULT_VAD_HOP_SECONDS))
        min_speech_seconds = float(vad_payload.get("min_speech_seconds", DEFAULT_VAD_MIN_SPEECH_SECONDS))
        min_silence_seconds = float(vad_payload.get("min_silence_seconds", DEFAULT_VAD_MIN_SILENCE_SECONDS))

        if mode in {"asr", "asr_vad"}:
            await self.ensure_model_loaded()
        if mode in {"vad", "asr_vad"}:
            await self.ensure_vad_model_loaded()
        if enable_diarization:
            await self.ensure_diarization_model_loaded()

        started = time.perf_counter()
        self.active_http_requests += 1
        self._mark_activity()
        try:
            if media_path is None:
                media_path = await self._download_remote_media(audio_url)
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                self._convert_media_to_audio_array,
                media_path,
            )
            transcript = None
            vad_result = None
            diarization_result = None
            async with self.inference_lock:
                if mode in {"asr", "asr_vad"}:
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._transcribe_audio_array,
                        audio,
                    )
                if mode in {"vad", "asr_vad"}:
                    vad_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._compute_vad_result,
                        audio,
                        threshold,
                        window_seconds,
                        hop_seconds,
                        min_speech_seconds,
                        min_silence_seconds,
                    )
                if enable_diarization:
                    diarization_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._compute_diarization_result,
                        audio,
                    )
        except Exception as exc:
            logger.error("API inference error for {}: {}", audio_url, exc)
            logger.error(traceback.format_exc())
            return web.json_response(
                {"error": "inference_failed", "message": str(exc)},
                status=500,
            )
        finally:
            self.active_http_requests = max(0, self.active_http_requests - 1)
            self._mark_activity()
            if media_path is not None and media_path.exists():
                media_path.unlink()

        duration_ms = int((time.perf_counter() - started) * 1000)
        response: dict[str, Any] = {
            "mode": mode,
            "duration_ms": duration_ms,
            "sample_rate": self.sample_rate,
            "enable_diarization": enable_diarization,
            "input_mode": input_mode,
        }
        if audio_url:
            response["audio_url"] = audio_url
        if source_filename:
            response["source_filename"] = source_filename
        if source_type:
            response["source_type"] = source_type
        if source_size is not None:
            response["source_size"] = source_size
        if transcript is not None:
            response["text"] = transcript
            response["model"] = self.model_name_or_path
        if vad_result is not None:
            response["vad"] = vad_result
        if diarization_result is not None:
            response["diarization"] = diarization_result
        return web.json_response(response)

    async def api_transcribe_handler(self, request: web.Request) -> web.Response:
        """Download media from a URL and return ASR, VAD, or combined results."""
        return await self._api_inference_handler(request)

    async def api_vad_handler(self, request: web.Request) -> web.Response:
        """Download media from a URL and return only voice activity detection results."""
        return await self._api_inference_handler(request, forced_mode="vad")

    async def api_diarize_handler(self, request: web.Request) -> web.Response:
        """Download media from a URL and return only speaker diarization results."""
        return await self._api_inference_handler(
            request,
            forced_mode="diarize",
            force_diarization=True,
        )

    async def start(self):
        """Start the HTTP + WebSocket server."""
        if self.preload_model:
            self.load_model()

        logger.info(
            "Starting streaming ASR server on ws://{}:{} (preload_model={}, idle_unload_seconds={})",
            self.host,
            self.port,
            self.preload_model,
            self.idle_unload_seconds,
        )

        app = web.Application(
            middlewares=[self.cors_middleware],
            client_max_size=DEFAULT_MAX_REQUEST_SIZE_MB * 1024 * 1024,
        )
        app.router.add_get("/", self.root_handler)
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/ws", self.websocket_handler)
        app.router.add_get("/api", self.api_info_handler)
        app.router.add_post("/api/transcribe", self.api_transcribe_handler)
        app.router.add_post("/api/vad", self.api_vad_handler)
        app.router.add_post("/api/diarize", self.api_diarize_handler)
        app.router.add_route("OPTIONS", "/{path:.*}", lambda request: web.Response(status=204))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.idle_monitor_task = asyncio.create_task(self.idle_monitor())

        logger.info("ASR server listening on ws://{}:{}", self.host, self.port)
        logger.info("Health check available at http://{}:{}/health", self.host, self.port)
        try:
            await asyncio.Future()
        finally:
            if self.idle_monitor_task is not None:
                self.idle_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.idle_monitor_task


def main():
    parser = argparse.ArgumentParser(
        description="Nemotron Streaming ASR WebSocket Server"
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9090")))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to local .nemo file",
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=int(os.environ.get("RIGHT_CONTEXT", "1")),
        choices=[0, 1, 6, 13],
        help="Right context frames: 0=80ms, 1=160ms, 6=560ms, 13=1.12s latency",
    )
    parser.add_argument(
        "--idle-unload-seconds",
        type=int,
        default=DEFAULT_IDLE_UNLOAD_SECONDS,
        help="Unload the model after this many idle seconds; use 0 to disable",
    )
    parser.add_argument(
        "--idle-check-interval-seconds",
        type=int,
        default=DEFAULT_IDLE_CHECK_INTERVAL_SECONDS,
        help="How often to check for idle model unloading",
    )
    parser.add_argument(
        "--preload-model",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PRELOAD_MODEL,
        help="Preload the model at startup instead of waiting for the first request",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help="NeMo pretrained Sortformer diarization checkpoint name",
    )
    args = parser.parse_args()

    server = ASRServer(
        model=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
        idle_unload_seconds=args.idle_unload_seconds,
        idle_check_interval_seconds=args.idle_check_interval_seconds,
        preload_model=args.preload_model,
        diarization_model=args.diarization_model,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
