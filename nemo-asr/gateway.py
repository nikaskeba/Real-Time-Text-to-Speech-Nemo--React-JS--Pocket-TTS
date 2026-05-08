"""Lightweight Docker gateway for the host-native Nemo ASR worker."""

import argparse
import asyncio
import contextlib
import json
import os
from typing import Any, Optional

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from loguru import logger


DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "9090"))
DEFAULT_UPSTREAM_BASE = os.environ.get("ASR_UPSTREAM_BASE", "http://host.docker.internal:9091").rstrip("/")
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("ASR_UPSTREAM_TIMEOUT", "300"))
DEFAULT_MAX_REQUEST_SIZE_MB = float(os.environ.get("ASR_MAX_REQUEST_SIZE_MB", "50"))
DB_API_BASE = os.environ.get("DB_API_BASE", "http://mysql_api:3001").rstrip("/")
DB_API_PREFIX = os.environ.get("DB_API_PREFIX", "/db-api")
DB_API_TIMEOUT_SECONDS = float(os.environ.get("DB_API_TIMEOUT", "8.0"))
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "1") == "1"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class ASRGateway:
    def __init__(self, upstream_base: str):
        self.upstream_base = upstream_base.rstrip("/")

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

    def _upstream_url(self, request: web.Request) -> str:
        return f"{self.upstream_base}{request.rel_url.path}"

    @staticmethod
    def _filtered_request_headers(request: web.Request) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            lower_key = key.lower()
            if lower_key in HOP_BY_HOP_HEADERS or lower_key == "host":
                continue
            if lower_key in {"authorization", "x-web-key", "x-user-email"}:
                continue
            headers[key] = value
        return headers

    @staticmethod
    def _filtered_response_headers(headers: "web.LooseHeaders") -> dict[str, str]:
        filtered: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            filtered[key] = value
        return filtered

    async def root_handler(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "nemo-asr-gateway",
                "mode": "docker-gateway",
                "upstream": self.upstream_base,
                "routes": {
                    "health": "/health",
                    "websocket": "/ws",
                    "api": "/api",
                    "transcribe": "/api/transcribe",
                    "vad": "/api/vad",
                    "diarize": "/api/diarize",
                },
            }
        )

    async def proxy_http_handler(self, request: web.Request, require_auth: bool = False) -> web.Response:
        if require_auth:
            await self.require_user(request)

        body = await request.read()
        timeout = ClientTimeout(total=DEFAULT_UPSTREAM_TIMEOUT_SECONDS)
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.request(
                    method=request.method,
                    url=self._upstream_url(request),
                    params=request.query,
                    data=body if body else None,
                    headers=self._filtered_request_headers(request),
                ) as upstream_response:
                    response_body = await upstream_response.read()
                    return web.Response(
                        status=upstream_response.status,
                        headers=self._filtered_response_headers(upstream_response.headers),
                        body=response_body,
                    )
            except web.HTTPException:
                raise
            except Exception as exc:
                logger.error("Gateway HTTP proxy error for {}: {}", request.rel_url, exc)
                raise web.HTTPBadGateway(
                    text=json.dumps({"error": "upstream_unavailable", "message": str(exc)}),
                    content_type="application/json",
                ) from exc

    async def health_handler(self, request: web.Request) -> web.Response:
        return await self.proxy_http_handler(request, require_auth=False)

    async def api_handler(self, request: web.Request) -> web.Response:
        return await self.proxy_http_handler(request, require_auth=True)

    async def transcribe_handler(self, request: web.Request) -> web.Response:
        return await self.proxy_http_handler(request, require_auth=True)

    async def vad_handler(self, request: web.Request) -> web.Response:
        return await self.proxy_http_handler(request, require_auth=True)

    async def diarize_handler(self, request: web.Request) -> web.Response:
        return await self.proxy_http_handler(request, require_auth=True)

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        await self.require_user(request)

        timeout = ClientTimeout(total=DEFAULT_UPSTREAM_TIMEOUT_SECONDS)
        async with ClientSession(timeout=timeout) as session:
            try:
                upstream_ws_url = f"{self.upstream_base}/ws"
                if request.query_string:
                    upstream_ws_url = f"{upstream_ws_url}?{request.query_string}"
                upstream_ws = await session.ws_connect(
                    upstream_ws_url,
                    max_msg_size=10 * 1024 * 1024,
                )
            except Exception as exc:
                logger.error("Gateway websocket upstream connect failed: {}", exc)
                raise web.HTTPBadGateway(
                    text=json.dumps({"error": "upstream_unavailable", "message": str(exc)}),
                    content_type="application/json",
                ) from exc

            client_ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
            await client_ws.prepare(request)

            async def client_to_upstream():
                async for message in client_ws:
                    if message.type == WSMsgType.BINARY:
                        await upstream_ws.send_bytes(message.data)
                    elif message.type == WSMsgType.TEXT:
                        await upstream_ws.send_str(message.data)
                    elif message.type == WSMsgType.CLOSE:
                        await upstream_ws.close()
                        break
                    elif message.type == WSMsgType.ERROR:
                        break

            async def upstream_to_client():
                async for message in upstream_ws:
                    if message.type == WSMsgType.BINARY:
                        await client_ws.send_bytes(message.data)
                    elif message.type == WSMsgType.TEXT:
                        await client_ws.send_str(message.data)
                    elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                        await client_ws.close()
                        break
                    elif message.type == WSMsgType.ERROR:
                        break

            try:
                relay_tasks = [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ]
                done, pending = await asyncio.wait(
                    relay_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            except Exception as exc:
                logger.error("Gateway websocket relay error: {}", exc)
            finally:
                await upstream_ws.close()
                await client_ws.close()

            return client_ws


def build_app(upstream_base: str) -> web.Application:
    gateway = ASRGateway(upstream_base)
    app = web.Application(client_max_size=DEFAULT_MAX_REQUEST_SIZE_MB * 1024 * 1024)
    app.router.add_get("/", gateway.root_handler)
    app.router.add_get("/health", gateway.health_handler)
    app.router.add_get("/api", gateway.api_handler)
    app.router.add_post("/api/transcribe", gateway.transcribe_handler)
    app.router.add_post("/api/vad", gateway.vad_handler)
    app.router.add_post("/api/diarize", gateway.diarize_handler)
    app.router.add_get("/ws", gateway.websocket_handler)
    return app


def main():
    parser = argparse.ArgumentParser(description="Docker gateway for host-native Nemo ASR")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream-base", default=DEFAULT_UPSTREAM_BASE)
    args = parser.parse_args()

    logger.info(
        "Starting ASR gateway on http://{}:{} -> {}",
        args.host,
        args.port,
        args.upstream_base,
    )
    web.run_app(build_app(args.upstream_base), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
