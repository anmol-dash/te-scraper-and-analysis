$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Dest = Join-Path $Root "src-tauri\binaries"

$rustc = & rustc -vV 2>$null
$triple = ($rustc | Select-String -Pattern '^host:\s*(.+)$').Matches.Groups[1].Value
if (-not $triple) {
  Write-Error "Could not read host triple from 'rustc -vV' (expected line starting with 'host:')"
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Push-Location $Backend
try {
  # Install bundle deps (certifi, paramiko + crypto stack) so PyInstaller can
  # collect them. Windows is single-arch (win_amd64 wheels) — no universal2
  # handling needed.
  python -m pip install -r requirements.txt
  python -m PyInstaller --noconfirm pyinstaller.spec
  $exe = Join-Path "dist" "pytool.exe"
  if (Test-Path $exe) {
    Copy-Item -Force $exe (Join-Path $Dest "pytool-$triple.exe")
    Write-Host "Installed $(Join-Path $Dest "pytool-$triple.exe")"
  } else {
    $elf = Join-Path "dist" "pytool"
    Copy-Item -Force $elf (Join-Path $Dest "pytool-$triple")
    Write-Host "Installed $(Join-Path $Dest "pytool-$triple")"
  }
} finally {
  Pop-Location
}
