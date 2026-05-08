# ASR Websocket React App

Basic React app for streaming microphone audio to an ASR websocket and rendering transcript messages.

Browsers cannot set custom `Authorization` headers on websocket handshakes, so this app includes a tiny local proxy. The React app connects to the local proxy, and the proxy connects to:

```http
GET /ws HTTP/1.1
Host: asr.skeba.info
Upgrade: websocket
Connection: Upgrade
Authorization: Bearer 123::your-api-key
```

## Run

Double-click the launcher for your OS:

- macOS: `Launch ASR.command`
- Windows: `Launch ASR.bat`

Or run manually:

```bash
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

The proxy listens at `ws://127.0.0.1:8787/ws` and forwards to `wss://asr.skeba.info/ws`.

## Configure

Edit `.env`:

```dotenv
ASR_WS_ENDPOINT=wss://asr.skeba.info/ws
ASR_AUTHORIZATION=Bearer 123::your-api-key
PROXY_PORT=8787
VITE_PROXY_WS_URL=ws://127.0.0.1:8787/ws
```

You can enter only the token value (`123::your-api-key`) or the full value (`Bearer 123::your-api-key`) for `ASR_AUTHORIZATION`. The proxy normalizes it before adding the `Authorization` header.
