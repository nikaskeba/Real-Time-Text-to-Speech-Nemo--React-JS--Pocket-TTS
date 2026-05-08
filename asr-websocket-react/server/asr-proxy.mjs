import { createServer } from "node:http";
import { once } from "node:events";
import { timingSafeEqual } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { extname, resolve } from "node:path";
import { WebSocket, WebSocketServer } from "ws";

const loadDotEnv = () => {
  const envPath = resolve(process.cwd(), ".env");
  if (!existsSync(envPath)) {
    return;
  }

  const lines = readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      continue;
    }

    const [key, ...valueParts] = trimmed.split("=");
    if (process.env[key]) {
      continue;
    }

    const rawValue = valueParts.join("=").trim();
    process.env[key] = rawValue.replace(/^['"]|['"]$/g, "");
  }
};

loadDotEnv();

const HOST = process.env.HOST ?? "0.0.0.0";
const PORT = Number(process.env.PORT ?? process.env.PROXY_PORT ?? 9090);
const UPSTREAM_WS_URL =
  process.env.ASR_WS_ENDPOINT ??
  process.env.UPSTREAM_WS_URL ??
  "ws://host.docker.internal:9091/ws";
const ASR_API_BASE_URL = process.env.ASR_API_BASE_URL ?? "http://host.docker.internal:9091";
const ASR_TRANSCRIBE_URL =
  process.env.ASR_TRANSCRIBE_URL ?? new URL("/api/transcribe", ASR_API_BASE_URL).toString();
const ASR_DIARIZE_URL =
  process.env.ASR_DIARIZE_URL ?? new URL("/api/diarize", ASR_API_BASE_URL).toString();
const ASR_HEALTH_URL =
  process.env.ASR_HEALTH_URL ?? new URL("/health", ASR_API_BASE_URL).toString();
const TTS_API_BASE_URL = process.env.TTS_API_BASE_URL ?? "http://host.docker.internal:9081";
const TTS_SYNTHESIZE_URL =
  process.env.TTS_SYNTHESIZE_URL ?? new URL("/api/synthesize", TTS_API_BASE_URL).toString();
const TTS_SYNTHESIZE_STREAM_URL =
  process.env.TTS_SYNTHESIZE_STREAM_URL ??
  new URL("/api/synthesize/stream", TTS_API_BASE_URL).toString();
const TTS_VOICES_URL =
  process.env.TTS_VOICES_URL ?? new URL("/api/voices", TTS_API_BASE_URL).toString();
const TTS_VOICE_CLONE_URL =
  process.env.TTS_VOICE_CLONE_URL ?? new URL("/api/voices/clone", TTS_API_BASE_URL).toString();
const TTS_HEALTH_URL =
  process.env.TTS_HEALTH_URL ?? new URL("/health", TTS_API_BASE_URL).toString();
const LLM_API_BASE_URL = process.env.LLM_API_BASE_URL ?? "http://host.docker.internal:5001";
const LLM_CHAT_COMPLETIONS_URL =
  process.env.LLM_CHAT_COMPLETIONS_URL ??
  new URL("/v1/chat/completions", LLM_API_BASE_URL).toString();
const LLM_MODELS_URL =
  process.env.LLM_MODELS_URL ?? new URL("/v1/models", LLM_API_BASE_URL).toString();
const DEFAULT_LLM_MODEL = process.env.LLM_MODEL ?? "gemma-3-4b-it";
const MESSAGE_LOG_LIMIT = Number(process.env.MESSAGE_LOG_LIMIT ?? 6);
const DIST_DIR = resolve(process.cwd(), "dist");
const FRONTEND_AUTH_USER =
  process.env.REACT_APP_AUTH_USER ?? process.env.FRONTEND_AUTH_USER ?? "";
const FRONTEND_AUTH_PASSWORD =
  process.env.REACT_APP_AUTH_PASSWORD ?? process.env.FRONTEND_AUTH_PASSWORD ?? "";
const FRONTEND_AUTH_REALM = process.env.FRONTEND_AUTH_REALM ?? "ASR";

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

const corsHeadersForRequest = (request) => {
  const origin = request.headers.origin ?? "*";
  return {
    "access-control-allow-headers": "content-type, authorization, x-api-key, x-web-key, x-user-email",
    "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
    "access-control-allow-origin": origin,
    "access-control-expose-headers":
      "content-type,x-tts-stream-format,x-audio-format,x-audio-channels,x-audio-sample-rate",
    vary: "Origin",
  };
};

const sendJson = (requestOrResponse, responseOrStatusCode, statusCodeOrPayload, maybePayload) => {
  const hasRequest = typeof maybePayload !== "undefined";
  const request = hasRequest ? requestOrResponse : null;
  const response = hasRequest ? responseOrStatusCode : requestOrResponse;
  const statusCode = hasRequest ? statusCodeOrPayload : responseOrStatusCode;
  const payload = hasRequest ? maybePayload : statusCodeOrPayload;
  response.writeHead(statusCode, {
    ...corsHeadersForRequest(request ?? { headers: {} }),
    "content-type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload));
};

const safeEqual = (left, right) => {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
};

const isFrontendAuthEnabled = () =>
  Boolean(FRONTEND_AUTH_USER.trim() && FRONTEND_AUTH_PASSWORD.trim());

const isFrontendAuthorized = (request) => {
  if (!isFrontendAuthEnabled()) {
    return true;
  }

  const authorization = request.headers.authorization ?? "";
  if (!authorization.toLowerCase().startsWith("basic ")) {
    return false;
  }

  try {
    const encoded = authorization.split(" ", 2)[1] ?? "";
    const decoded = Buffer.from(encoded, "base64").toString("utf8");
    const separatorIndex = decoded.indexOf(":");
    if (separatorIndex === -1) {
      return false;
    }

    const username = decoded.slice(0, separatorIndex);
    const password = decoded.slice(separatorIndex + 1);
    return (
      safeEqual(username, FRONTEND_AUTH_USER) &&
      safeEqual(password, FRONTEND_AUTH_PASSWORD)
    );
  } catch {
    return false;
  }
};

const requestFrontendAuth = (response) => {
  response.writeHead(401, {
    "cache-control": "no-store",
    "content-type": "text/plain; charset=utf-8",
    "www-authenticate": `Basic realm="${FRONTEND_AUTH_REALM}", charset="UTF-8"`,
  });
  response.end("Authentication required");
};

const noBufferStreamHeaders = {
  "cache-control": "no-cache, no-transform",
  "connection": "keep-alive",
  "transfer-encoding": "chunked",
  "x-accel-buffering": "no",
};

const prepareNoBufferStream = (response) => {
  response.socket?.setNoDelay?.(true);
  response.socket?.setKeepAlive?.(true);
  response.flushHeaders?.();
};

const writeStreamChunk = async (response, chunk) => {
  if (!response.write(Buffer.from(chunk))) {
    await once(response, "drain");
  }
};

const readJsonBody = async (request) => {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const body = Buffer.concat(chunks).toString("utf8");
  return body ? JSON.parse(body) : {};
};

const readRequestBuffer = async (request) => {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
};

const sendStaticFile = (request, response, filePath) => {
  if (!existsSync(filePath)) {
    return false;
  }

  const extension = extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[extension] ?? "application/octet-stream";
  response.writeHead(200, {
    ...corsHeadersForRequest(request),
    "cache-control": extension === ".html" ? "no-cache" : "public, max-age=31536000, immutable",
    "content-type": contentType,
  });
  response.end(readFileSync(filePath));
  return true;
};

const serveFrontend = (request, response, requestUrl) => {
  if (!isFrontendAuthorized(request)) {
    requestFrontendAuth(response);
    return;
  }

  const relativePath = requestUrl.pathname === "/" ? "/index.html" : requestUrl.pathname;
  const safePath = relativePath.replace(/^\/+/, "").replace(/\.\./g, "");
  const filePath = resolve(DIST_DIR, safePath);
  if (sendStaticFile(request, response, filePath)) {
    return;
  }

  const indexPath = resolve(DIST_DIR, "index.html");
  if (sendStaticFile(request, response, indexPath)) {
    return;
  }

  sendJson(request, response, 404, {
    ok: false,
    error: "frontend_not_built",
    message: "React dist assets are missing from the ASR proxy container.",
  });
};

const server = createServer((request, response) => {
  if (request.method === "OPTIONS") {
    response.writeHead(204, corsHeadersForRequest(request));
    response.end();
    return;
  }

  const requestUrl = new URL(request.url ?? "/", `http://${request.headers.host}`);

  if (requestUrl.pathname === "/health" && request.method === "GET") {
    void handleHealth(request, response);
    return;
  }

  if (requestUrl.pathname === "/api" && request.method === "GET") {
    sendJson(request, response, 200, {
      ok: true,
      mode: "asr-web-app-proxy",
      websocket: "/ws",
      frontend: "/",
      upstream: UPSTREAM_WS_URL,
      asrApiBaseUrl: ASR_API_BASE_URL,
      asrTranscribe: "/api/transcribe",
      asrDiarize: "/api/diarize",
      ttsVoices: "/api/tts/voices",
      ttsSynthesize: "/api/tts/synthesize",
      ttsSynthesizeStream: "/api/tts/synthesize/stream",
      ttsCloneVoice: "/api/tts/voices/clone",
      chatModels: "/api/chat/models",
      chatCompletions: "/api/chat/completions",
      defaultLlmModel: DEFAULT_LLM_MODEL,
      usesLocalAsr: ASR_API_BASE_URL.includes("host.docker.internal") || ASR_API_BASE_URL.includes("127.0.0.1"),
      usesLocalTts: TTS_API_BASE_URL.includes("host.docker.internal") || TTS_API_BASE_URL.includes("127.0.0.1"),
      usesLocalLlm: LLM_API_BASE_URL.includes("host.docker.internal") || LLM_API_BASE_URL.includes("127.0.0.1"),
    });
    return;
  }

  if (requestUrl.pathname === "/api/transcribe" && request.method === "POST") {
    void handleTranscribe(request, response);
    return;
  }

  if (requestUrl.pathname === "/api/diarize" && request.method === "POST") {
    void handleDiarize(request, response);
    return;
  }

  if (requestUrl.pathname === "/api/tts/voices" && request.method === "GET") {
    void handleTtsVoices(response);
    return;
  }

  if (requestUrl.pathname === "/api/tts/synthesize" && request.method === "POST") {
    void handleTtsSynthesize(request, response);
    return;
  }

  if (requestUrl.pathname === "/api/tts/synthesize/stream" && request.method === "POST") {
    void handleTtsSynthesizeStream(request, response);
    return;
  }

  if (requestUrl.pathname === "/api/tts/voices/clone" && request.method === "POST") {
    void handleTtsCloneVoice(request, response);
    return;
  }

  if (requestUrl.pathname.startsWith("/api/tts/voices/clone/") && request.method === "DELETE") {
    void handleTtsDeleteClone(requestUrl, response);
    return;
  }

  if (requestUrl.pathname === "/api/chat/completions" && request.method === "POST") {
    void handleChatCompletions(request, response);
    return;
  }

  if (requestUrl.pathname === "/api/chat/models" && request.method === "GET") {
    void handleChatModels(response);
    return;
  }

  serveFrontend(request, response, requestUrl);
});

const wss = new WebSocketServer({ server, path: "/ws" });

const asBearerValue = (authorization) => {
  const value = authorization?.trim() ?? "";
  if (!value) {
    return "";
  }
  return value.toLowerCase().startsWith("bearer ") ? value : `Bearer ${value}`;
};

const withAuthorizationHeader = (authorization, headers = {}) =>
  authorization ? { ...headers, Authorization: authorization } : headers;

const getAsrAuthorization = () =>
  asBearerValue(process.env.ASR_AUTHORIZATION ?? process.env.ASR_API_KEY);

const getLlmAuthorization = () =>
  asBearerValue(
    process.env.LLM_AUTHORIZATION ?? process.env.ASR_AUTHORIZATION ?? process.env.ASR_API_KEY,
  );

const getTtsAuthorization = () =>
  asBearerValue(process.env.TTS_AUTHORIZATION ?? process.env.ASR_AUTHORIZATION ?? process.env.ASR_API_KEY);

const buildUpstreamWsUrl = (clientUrl) => {
  const upstreamUrl = new URL(UPSTREAM_WS_URL);
  for (const [key, value] of clientUrl.searchParams) {
    upstreamUrl.searchParams.set(key, value);
  }
  return upstreamUrl.toString();
};

const closePair = (client, upstream, code = 1000, reason = "closed") => {
  const closeCode = code && ![1005, 1006, 1015].includes(code) ? code : 1000;
  if (client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING) {
    client.close(closeCode, reason);
  }
  if (upstream.readyState === WebSocket.OPEN || upstream.readyState === WebSocket.CONNECTING) {
    upstream.close(closeCode, reason);
  }
};

const handleHealth = async (request, response) => {
  try {
    const asrResponse = await fetch(ASR_HEALTH_URL);
    const asrPayload = await asrResponse.json();
    let ttsPayload = null;
    try {
      const ttsResponse = await fetch(TTS_HEALTH_URL);
      ttsPayload = await ttsResponse.json();
    } catch {
      ttsPayload = null;
    }
    sendJson(request, response, asrResponse.ok ? 200 : asrResponse.status, {
      status: asrResponse.ok ? "ok" : "degraded",
      frontend_served: true,
      asr: asrPayload,
      tts: ttsPayload,
      upstreams: {
        asr: ASR_API_BASE_URL,
        tts: TTS_API_BASE_URL,
        llm: LLM_API_BASE_URL,
      },
    });
  } catch (error) {
    sendJson(request, response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "Health request failed",
    });
  }
};

const handleTranscribe = async (request, response) => {
  const authorization = getAsrAuthorization();

  try {
    const contentType = request.headers["content-type"] ?? "";
    if (contentType.includes("multipart/form-data")) {
      const body = await readRequestBuffer(request);
      const upstreamResponse = await fetch(ASR_TRANSCRIBE_URL, {
        method: "POST",
        headers: withAuthorizationHeader(authorization, {
          "Content-Type": contentType,
        }),
        body,
      });
      const { payload } = await readUpstreamPayload(upstreamResponse);

      sendJson(request, response, upstreamResponse.status, {
        ok: upstreamResponse.ok,
        status: upstreamResponse.status,
        statusText: upstreamResponse.statusText,
        upstream: ASR_TRANSCRIBE_URL,
        result: payload,
      });
      return;
    }

    const body = await readJsonBody(request);
    if (!body.audio_url || typeof body.audio_url !== "string") {
      sendJson(response, 400, {
        error: "audio_url is required for JSON requests, or send multipart/form-data with an audio file",
      });
      return;
    }

    const upstreamResponse = await fetch(ASR_TRANSCRIBE_URL, {
      method: "POST",
      headers: withAuthorizationHeader(authorization, {
        "Content-Type": "application/json",
      }),
      body: JSON.stringify({
        audio_url: body.audio_url,
        ...(body.mode ? { mode: body.mode } : {}),
        ...(body.enable_diarization != null
          ? { enable_diarization: Boolean(body.enable_diarization) }
          : {}),
        ...(body.diarization_speakers ? { diarization_speakers: body.diarization_speakers } : {}),
        ...(body.num_speakers ? { num_speakers: body.num_speakers } : {}),
        ...(body.min_speakers ? { min_speakers: body.min_speakers } : {}),
        ...(body.max_speakers ? { max_speakers: body.max_speakers } : {}),
      }),
    });

    const text = await upstreamResponse.text();
    let payload;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { text };
    }

    sendJson(request, response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: ASR_TRANSCRIBE_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(request, response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "Transcription request failed",
      upstream: ASR_TRANSCRIBE_URL,
    });
  }
};

const handleDiarize = async (request, response) => {
  const authorization = getAsrAuthorization();

  try {
    const contentType = request.headers["content-type"] ?? "";
    if (contentType.includes("multipart/form-data")) {
      const body = await readRequestBuffer(request);
      const upstreamResponse = await fetch(ASR_DIARIZE_URL, {
        method: "POST",
        headers: withAuthorizationHeader(authorization, {
          "Content-Type": contentType,
        }),
        body,
      });
      const { payload } = await readUpstreamPayload(upstreamResponse);

      sendJson(request, response, upstreamResponse.status, {
        ok: upstreamResponse.ok,
        status: upstreamResponse.status,
        statusText: upstreamResponse.statusText,
        upstream: ASR_DIARIZE_URL,
        result: payload,
      });
      return;
    }

    const body = await readJsonBody(request);
    if (!body.audio_url || typeof body.audio_url !== "string") {
      sendJson(request, response, 400, {
        error: "audio_url is required for JSON requests, or send multipart/form-data with an audio file",
      });
      return;
    }

    const upstreamResponse = await fetch(ASR_DIARIZE_URL, {
      method: "POST",
      headers: withAuthorizationHeader(authorization, {
        "Content-Type": "application/json",
      }),
      body: JSON.stringify({
        audio_url: body.audio_url,
        ...(body.enable_diarization != null
          ? { enable_diarization: Boolean(body.enable_diarization) }
          : {}),
        ...(body.diarization_speakers ? { diarization_speakers: body.diarization_speakers } : {}),
        ...(body.num_speakers ? { num_speakers: body.num_speakers } : {}),
        ...(body.min_speakers ? { min_speakers: body.min_speakers } : {}),
        ...(body.max_speakers ? { max_speakers: body.max_speakers } : {}),
      }),
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(request, response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: ASR_DIARIZE_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(request, response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "Diarization request failed",
      upstream: ASR_DIARIZE_URL,
    });
  }
};

const readUpstreamPayload = async (upstreamResponse) => {
  const contentType = upstreamResponse.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const text = await upstreamResponse.text();
    try {
      return {
        payload: text ? JSON.parse(text) : {},
        rawText: text,
      };
    } catch {
      return {
        payload: { text },
        rawText: text,
      };
    }
  }

  if (contentType.startsWith("audio/") || contentType.includes("octet-stream")) {
    const buffer = Buffer.from(await upstreamResponse.arrayBuffer());
    return {
      payload: {
        audioBase64: buffer.toString("base64"),
        mimeType: contentType || "audio/wav",
      },
      rawText: "",
    };
  }

  const text = await upstreamResponse.text();
  try {
    return {
      payload: text ? JSON.parse(text) : {},
      rawText: text,
    };
  } catch {
    return {
      payload: { text },
      rawText: text,
    };
  }
};

const handleTtsVoices = async (response) => {
  const authorization = getTtsAuthorization();

  try {
    const upstreamResponse = await fetch(TTS_VOICES_URL, {
      headers: withAuthorizationHeader(authorization),
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: TTS_VOICES_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "TTS voices request failed",
      upstream: TTS_VOICES_URL,
    });
  }
};

const handleTtsSynthesize = async (request, response) => {
  const authorization = getTtsAuthorization();

  try {
    const body = await readJsonBody(request);
    const text = body.text?.trim?.() ?? "";
    if (!text) {
      sendJson(response, 400, {
        error: "text is required",
      });
      return;
    }

    const upstreamResponse = await fetch(TTS_SYNTHESIZE_URL, {
      method: "POST",
      headers: withAuthorizationHeader(authorization, {
        "Content-Type": "application/json",
      }),
      body: JSON.stringify({
        text,
        voice: body.voice || "alba",
        language: body.language || "english",
        response_format: body.response_format || "base64",
      }),
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: TTS_SYNTHESIZE_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "TTS synthesize request failed",
      upstream: TTS_SYNTHESIZE_URL,
    });
  }
};

const handleTtsSynthesizeStream = async (request, response) => {
  const authorization = getTtsAuthorization();

  try {
    const body = await readJsonBody(request);
    const text = body.text?.trim?.() ?? "";
    if (!text) {
      sendJson(response, 400, {
        error: "text is required",
      });
      return;
    }

    const upstreamResponse = await fetch(TTS_SYNTHESIZE_STREAM_URL, {
      method: "POST",
      headers: withAuthorizationHeader(authorization, {
        Accept: "application/octet-stream",
        "Content-Type": "application/json",
      }),
      body: JSON.stringify({
        text,
        voice: body.voice || "alba",
        language: body.language || "english",
        stream_format: body.stream_format || "pcm_s16le",
      }),
    });

    const contentType = upstreamResponse.headers.get("content-type") ?? "application/octet-stream";
    if (!upstreamResponse.ok) {
      const textBody = await upstreamResponse.text();
      let payload;
      try {
        payload = textBody ? JSON.parse(textBody) : {};
      } catch {
        payload = { text: textBody };
      }
      sendJson(response, upstreamResponse.status, {
        ok: false,
        status: upstreamResponse.status,
        statusText: upstreamResponse.statusText,
        upstream: TTS_SYNTHESIZE_STREAM_URL,
        result: payload,
      });
      return;
    }

    response.writeHead(upstreamResponse.status, {
      ...noBufferStreamHeaders,
      ...corsHeadersForRequest(request),
      "access-control-expose-headers":
        "content-type,x-tts-stream-format,x-audio-format,x-audio-channels,x-audio-sample-rate",
      "content-type": contentType,
      "x-tts-stream-format": upstreamResponse.headers.get("x-tts-stream-format") ?? "",
      "x-audio-format": upstreamResponse.headers.get("x-audio-format") ?? "",
      "x-audio-channels": upstreamResponse.headers.get("x-audio-channels") ?? "",
      "x-audio-sample-rate": upstreamResponse.headers.get("x-audio-sample-rate") ?? "",
    });
    prepareNoBufferStream(response);

    if (!upstreamResponse.body) {
      response.end();
      return;
    }

    for await (const chunk of upstreamResponse.body) {
      await writeStreamChunk(response, chunk);
    }
    response.end();
  } catch (error) {
    if (!response.headersSent) {
      sendJson(response, 502, {
        ok: false,
        error: error instanceof Error ? error.message : "TTS stream request failed",
        upstream: TTS_SYNTHESIZE_STREAM_URL,
      });
    } else {
      response.end();
    }
  }
};

const handleTtsCloneVoice = async (request, response) => {
  const authorization = getTtsAuthorization();

  const contentType = request.headers["content-type"];
  if (!contentType?.includes("multipart/form-data")) {
    sendJson(response, 400, {
      error: "multipart/form-data with audio, clone_id, and language fields is required",
    });
    return;
  }

  try {
    const body = await readRequestBuffer(request);
    const upstreamResponse = await fetch(TTS_VOICE_CLONE_URL, {
      method: "POST",
      headers: withAuthorizationHeader(authorization, {
        "Content-Type": contentType,
      }),
      body,
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: TTS_VOICE_CLONE_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "TTS voice clone request failed",
      upstream: TTS_VOICE_CLONE_URL,
    });
  }
};

const handleTtsDeleteClone = async (requestUrl, response) => {
  const authorization = getTtsAuthorization();

  const cloneId = decodeURIComponent(requestUrl.pathname.split("/").pop() ?? "").trim();
  if (!cloneId) {
    sendJson(response, 400, {
      error: "clone_id is required",
    });
    return;
  }

  const upstreamUrl = `${TTS_VOICE_CLONE_URL.replace(/\/$/, "")}/${encodeURIComponent(cloneId)}`;
  try {
    const upstreamResponse = await fetch(upstreamUrl, {
      method: "DELETE",
      headers: withAuthorizationHeader(authorization),
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: upstreamUrl,
      result: payload,
    });
  } catch (error) {
    sendJson(response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "TTS voice delete request failed",
      upstream: upstreamUrl,
    });
  }
};

const handleChatModels = async (response) => {
  const authorization = getLlmAuthorization();
  if (!authorization) {
    sendJson(response, 500, {
      error: "Missing LLM_AUTHORIZATION or ASR_AUTHORIZATION in .env",
    });
    return;
  }

  try {
    const upstreamResponse = await fetch(LLM_MODELS_URL, {
      headers: {
        Authorization: authorization,
      },
    });
    const { payload } = await readUpstreamPayload(upstreamResponse);

    sendJson(response, upstreamResponse.status, {
      ok: upstreamResponse.ok,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      upstream: LLM_MODELS_URL,
      result: payload,
    });
  } catch (error) {
    sendJson(response, 502, {
      ok: false,
      error: error instanceof Error ? error.message : "Chat models request failed",
      upstream: LLM_MODELS_URL,
    });
  }
};

const handleChatCompletions = async (request, response) => {
  const authorization = getLlmAuthorization();
  if (!authorization) {
    sendJson(response, 500, {
      error: "Missing LLM_AUTHORIZATION or ASR_AUTHORIZATION in .env",
    });
    return;
  }

  try {
    const body = await readJsonBody(request);
    if (!Array.isArray(body.messages) || !body.messages.length) {
      sendJson(response, 400, {
        error: "messages array is required",
      });
      return;
    }

    const maxTokens = Number(body.max_tokens ?? body.maxTokens);
    const hasMaxTokens = Number.isFinite(maxTokens) && maxTokens > 0;
    const contextWindow = Number(body.context_window ?? body.contextWindow);
    const hasContextWindow = Number.isFinite(contextWindow) && contextWindow > 0;

    console.log(
      `Chat completion request: model=${body.model || DEFAULT_LLM_MODEL}, messages=${body.messages.length}, stream=${body.stream ?? true}, max_tokens=${hasMaxTokens ? Math.floor(maxTokens) : "none"}, context_window=${hasContextWindow ? Math.floor(contextWindow) : "none"}`,
    );

    const upstreamResponse = await fetch(LLM_CHAT_COMPLETIONS_URL, {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        Authorization: authorization,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: body.model || DEFAULT_LLM_MODEL,
        messages: body.messages,
        stream: body.stream ?? true,
        temperature: body.temperature ?? 0.7,
        ...(hasMaxTokens ? { max_tokens: Math.floor(maxTokens) } : {}),
        ...(hasContextWindow ? { context_window: Math.floor(contextWindow) } : {}),
      }),
    });

    const contentType = upstreamResponse.headers.get("content-type") ?? "text/event-stream";
    if (!upstreamResponse.ok) {
      const textBody = await upstreamResponse.text();
      let payload;
      try {
        payload = textBody ? JSON.parse(textBody) : {};
      } catch {
        payload = { text: textBody };
      }
      sendJson(response, upstreamResponse.status, {
        ok: false,
        status: upstreamResponse.status,
        statusText: upstreamResponse.statusText,
        upstream: LLM_CHAT_COMPLETIONS_URL,
        result: payload,
      });
      return;
    }

    response.writeHead(upstreamResponse.status, {
      ...noBufferStreamHeaders,
      ...corsHeadersForRequest(request),
      "access-control-expose-headers": "content-type",
      "content-type": contentType,
    });
    prepareNoBufferStream(response);

    if (!upstreamResponse.body) {
      response.end();
      return;
    }

    for await (const chunk of upstreamResponse.body) {
      await writeStreamChunk(response, chunk);
    }
    response.end();
  } catch (error) {
    if (!response.headersSent) {
      sendJson(response, 502, {
        ok: false,
        error: error instanceof Error ? error.message : "Chat completion request failed",
        upstream: LLM_CHAT_COMPLETIONS_URL,
      });
    } else {
      response.end();
    }
  }
};

wss.on("connection", (client, request) => {
  const url = new URL(request.url ?? "/ws", `http://${request.headers.host}`);
  const upstreamWsUrl = buildUpstreamWsUrl(url);
  const authorization = asBearerValue(
    process.env.ASR_AUTHORIZATION ??
      process.env.ASR_API_KEY ??
      url.searchParams.get("authorization") ??
      "",
  );

  let upstreamMessageCount = 0;
  let clientChunkCount = 0;
  let clientByteCount = 0;

  const upstream = new WebSocket(upstreamWsUrl, {
    headers: withAuthorizationHeader(authorization),
  });

  upstream.on("open", () => {
    client.send(
      JSON.stringify({
        type: "proxy-status",
        status: "connected",
        upstream: upstreamWsUrl,
      }),
    );
  });

  client.on("message", (data, isBinary) => {
    clientChunkCount += 1;
    clientByteCount += data.length ?? data.byteLength ?? 0;
    if (clientChunkCount <= MESSAGE_LOG_LIMIT || clientChunkCount % 20 === 0) {
      console.log(
        `Client audio chunk #${clientChunkCount}: ${data.length ?? data.byteLength ?? 0} bytes`,
      );
    }

    if (upstream.readyState === WebSocket.OPEN) {
      upstream.send(data, { binary: isBinary });
    }
  });

  upstream.on("message", (data, isBinary) => {
    upstreamMessageCount += 1;
    const text = isBinary ? "" : data.toString("utf8");
    if (upstreamMessageCount <= MESSAGE_LOG_LIMIT || upstreamMessageCount % 20 === 0) {
      const preview = isBinary ? `<binary ${data.length ?? data.byteLength ?? 0} bytes>` : text.slice(0, 240);
      console.log(`Upstream message #${upstreamMessageCount}: ${preview}`);
    }

    if (client.readyState === WebSocket.OPEN) {
      client.send(isBinary ? data : text, { binary: isBinary });
    }
  });

  client.on("close", (code, buffer) => {
    console.log(
      `Client closed: ${code} ${buffer.toString()} after ${clientChunkCount} chunks / ${clientByteCount} bytes`,
    );
    closePair(client, upstream, code, buffer.toString());
  });
  upstream.on("close", (code, buffer) => {
    const reason = buffer.toString();
    console.log(
      `Upstream closed: ${code} ${reason} after ${upstreamMessageCount} messages`,
    );
    if (client.readyState === WebSocket.OPEN && upstreamMessageCount === 0) {
      client.send(
        JSON.stringify({
          type: "proxy-status",
          status: "upstream-closed-without-transcript",
          code,
          reason,
          message: "ASR upstream closed before sending transcript messages.",
        }),
      );
    }
    closePair(client, upstream, code, reason);
  });

  upstream.on("unexpected-response", (_request, response) => {
    const chunks = [];
    response.on("data", (chunk) => chunks.push(chunk));
    response.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8").slice(0, 500);
      const message = `Upstream rejected websocket handshake: ${response.statusCode} ${response.statusMessage}`;
      console.error(`${message}${body ? ` - ${body}` : ""}`);

      if (client.readyState === WebSocket.OPEN) {
        client.send(
          JSON.stringify({
            type: "proxy-error",
            message,
            statusCode: response.statusCode,
            statusMessage: response.statusMessage,
            server: response.headers.server,
            body,
          }),
        );
      }
      closePair(client, upstream, 1011, message);
    });
  });

  client.on("error", (error) => {
    console.error("Client websocket error:", error.message);
    closePair(client, upstream, 1011, "Client websocket error");
  });

  upstream.on("error", (error) => {
    console.error("Upstream websocket error:", error.message);
    if (client.readyState === WebSocket.OPEN) {
      client.send(
        JSON.stringify({
          type: "proxy-error",
          message: error.message,
        }),
      );
    }
    closePair(client, upstream, 1011, "Upstream websocket error");
  });
});

server.listen(PORT, HOST, () => {
  console.log(`ASR web app proxy listening at http://${HOST}:${PORT}`);
  console.log(`Frontend root served from ${DIST_DIR}`);
  console.log(`ASR upstream websocket: ${UPSTREAM_WS_URL}`);
  console.log(`ASR upstream HTTP: ${ASR_API_BASE_URL}`);
  console.log(`TTS upstream HTTP: ${TTS_API_BASE_URL}`);
  console.log(`LLM upstream HTTP: ${LLM_API_BASE_URL}`);
  if (!getLlmAuthorization()) {
    console.warn("Missing LLM_AUTHORIZATION or ASR_AUTHORIZATION. Chat completions may fail against the local LLM proxy.");
  }
});
