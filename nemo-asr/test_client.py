"""Small local smoke test client for the Nemo ASR websocket server."""

import argparse
import asyncio
import json
import wave
from pathlib import Path

import websockets


def _read_pcm16_chunks(path: Path, chunk_ms: int):
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16000:
            raise ValueError("Expected 16kHz WAV input")
        if wav_file.getnchannels() != 1:
            raise ValueError("Expected mono WAV input")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV input")

        frames_per_chunk = int(16000 * (chunk_ms / 1000))
        while True:
            frames = wav_file.readframes(frames_per_chunk)
            if not frames:
                break
            yield frames


async def run_test(url: str, wav_path: Path, chunk_ms: int):
    async with websockets.connect(url, max_size=10 * 1024 * 1024) as websocket:
        ready = await websocket.recv()
        print("ready:", ready)

        for chunk in _read_pcm16_chunks(wav_path, chunk_ms):
            await websocket.send(chunk)
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.15)
                print("recv:", message)
            except asyncio.TimeoutError:
                pass

        await websocket.send(json.dumps({"type": "end", "finalize": True}))

        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2)
                print("recv:", message)
                parsed = json.loads(message)
                if parsed.get("message_type") == "committed_transcript":
                    break
            except asyncio.TimeoutError:
                break


def main():
    parser = argparse.ArgumentParser(description="Local Nemo ASR websocket smoke test")
    parser.add_argument("wav", type=Path, help="16kHz mono PCM16 WAV file")
    parser.add_argument("--url", default="ws://127.0.0.1:9090/ws")
    parser.add_argument("--chunk-ms", type=int, default=160)
    args = parser.parse_args()

    asyncio.run(run_test(args.url, args.wav, args.chunk_ms))


if __name__ == "__main__":
    main()
