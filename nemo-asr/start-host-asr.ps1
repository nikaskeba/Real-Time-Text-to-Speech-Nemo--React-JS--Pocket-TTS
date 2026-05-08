param(
  [string]$ModelPath = "C:\docker\voice\nemo-asr\models\nemotron-speech-streaming-en-0.6b.nemo",
  [int]$Port = 9091,
  [int]$RightContext = 1,
  [int]$IdleUnloadSeconds = 60,
  [int]$IdleCheckIntervalSeconds = 15,
  [switch]$PreloadModel
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".host-venv\Scripts\python.exe"
$ffmpegCandidates = @(
  "C:\Program Files\Krita (x64)\bin\ffmpeg.exe",
  "C:\pinokio\bin\miniconda\Library\bin\ffmpeg.exe",
  "C:\Users\nicho\Documents\ComfyUI\.venv\Scripts\ffmpeg.exe"
)
$ffmpegPath = $ffmpegCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

$env:HOST = "127.0.0.1"
$env:PORT = "$Port"
$env:NEMO_MODEL = $ModelPath
$env:RIGHT_CONTEXT = "$RightContext"
$env:IDLE_UNLOAD_SECONDS = "$IdleUnloadSeconds"
$env:IDLE_CHECK_INTERVAL_SECONDS = "$IdleCheckIntervalSeconds"
$env:PRELOAD_MODEL = if ($PreloadModel) { "1" } else { "0" }
$env:REQUIRE_AUTH = "0"
if ($ffmpegPath) {
  $env:FFMPEG_BIN = $ffmpegPath
  $ffmpegDir = Split-Path $ffmpegPath -Parent
  if ($ffmpegDir) {
    $env:PATH = "$ffmpegDir;$env:PATH"
  }
}

Write-Host "Starting host-native Nemo ASR on http://127.0.0.1:$Port"
Write-Host "Model: $ModelPath"
Write-Host "Preload model: $($env:PRELOAD_MODEL)"
Write-Host "Idle unload seconds: $IdleUnloadSeconds"
if ($env:FFMPEG_BIN) {
  Write-Host "FFmpeg: $($env:FFMPEG_BIN)"
}

if (Test-Path $venvPython) {
  & $venvPython .\server.py
} else {
  $pythonCommands = @(
    (Get-Command python -ErrorAction SilentlyContinue),
    (Get-Command py -ErrorAction SilentlyContinue),
    (Get-Command python3 -ErrorAction SilentlyContinue)
  ) | Where-Object { $_ }

  if (-not $pythonCommands) {
    throw "No Python launcher was found. Install Python or update PATH before starting the host ASR worker."
  }

  $pythonCommand = $pythonCommands[0].Name
  if ($pythonCommand -eq "py") {
    & $pythonCommand -3 .\server.py
  } else {
    & $pythonCommand .\server.py
  }
}
