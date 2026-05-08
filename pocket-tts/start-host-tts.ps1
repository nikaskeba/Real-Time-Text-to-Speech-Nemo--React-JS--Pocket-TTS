param(
  [int]$Port = 9081,
  [string]$Language = "english",
  [string]$DefaultVoice = "alba",
  [int]$IdleUnloadSeconds = 300,
  [int]$IdleCheckIntervalSeconds = 30,
  [switch]$PreloadModel
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".host-venv\Scripts\python.exe"

$env:HOST = "127.0.0.1"
$env:PORT = "$Port"
$env:TTS_LANGUAGE = $Language
$env:TTS_DEFAULT_VOICE = $DefaultVoice
$env:IDLE_UNLOAD_SECONDS = "$IdleUnloadSeconds"
$env:IDLE_CHECK_INTERVAL_SECONDS = "$IdleCheckIntervalSeconds"
$env:PRELOAD_MODEL = if ($PreloadModel) { "1" } else { "0" }
$env:REQUIRE_AUTH = "0"

Write-Host "Starting host-native Pocket TTS on http://127.0.0.1:$Port"
Write-Host "Language: $Language"
Write-Host "Default voice: $DefaultVoice"
Write-Host "Preload model: $($env:PRELOAD_MODEL)"
Write-Host "Idle unload seconds: $IdleUnloadSeconds"

if (Test-Path $venvPython) {
  & $venvPython .\app.py
} else {
  $pythonCommands = @(
    (Get-Command python -ErrorAction SilentlyContinue),
    (Get-Command py -ErrorAction SilentlyContinue),
    (Get-Command python3 -ErrorAction SilentlyContinue)
  ) | Where-Object { $_ }

  if (-not $pythonCommands) {
    throw "No Python launcher was found. Install Python or update PATH before starting the host TTS worker."
  }

  $pythonCommand = $pythonCommands[0].Name
  if ($pythonCommand -eq "py") {
    & $pythonCommand -3 .\app.py
  } else {
    & $pythonCommand .\app.py
  }
}
