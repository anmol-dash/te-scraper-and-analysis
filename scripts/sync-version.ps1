$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path "$PSScriptRoot\.."

python "$ROOT\scripts\_sync-version.py"
