#!/bin/zsh

set -e

cd "$(dirname "$0")"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required to run this app."
  echo "Install it from https://nodejs.org/ and run this launcher again."
  read -r "?Press Enter to close."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to run this app."
  read -r "?Press Enter to close."
  exit 1
fi

if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

echo "Starting ASR websocket app..."
echo "The browser will open at http://127.0.0.1:5173/"
echo "Press Ctrl+C in this window to stop the app."

(sleep 2 && open "http://127.0.0.1:5173/") &

npm run dev
