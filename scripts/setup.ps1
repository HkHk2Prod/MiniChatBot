<#
.SYNOPSIS
    Initialize a virtual environment for MiniChatBot using uv.

.DESCRIPTION
    Creates .venv with `uv venv` (if missing) and installs the project in
    editable mode with dev + tensorboard + data extras. torch is fetched via
    uv's --torch-backend, which auto-detects your CUDA driver by default and
    falls back to a CPU wheel when no NVIDIA GPU is present. Run from the
    project root. Requires uv: https://docs.astral.sh/uv/

.PARAMETER TorchBackend
    torch backend: auto (default), cpu, cu118, cu121, cu124, cu126, cu128.

.PARAMETER Cpu
    Shortcut for -TorchBackend cpu.

.PARAMETER NoExtras
    Skip the [dev,tensorboard,data] extras.

.PARAMETER Force
    Reinstall torch (e.g. switching CPU<->CUDA wheels).

.PARAMETER Python
    Python version or interpreter for the venv. Default: "python".

.EXAMPLE
    .\scripts\setup.ps1                          # auto-detect backend, full extras
    .\scripts\setup.ps1 -TorchBackend cu128      # force a CUDA build
    .\scripts\setup.ps1 -Cpu                     # CPU-only wheel
    .\scripts\setup.ps1 -Force                   # reinstall torch
#>

[CmdletBinding()]
param(
    [string]$TorchBackend = "auto",
    [switch]$Cpu,
    [switch]$NoExtras,
    [switch]$Force,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "pyproject.toml")) {
    Write-Error "pyproject.toml not found in $(Get-Location). Run this from the project root."
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it, then re-run: https://docs.astral.sh/uv/getting-started/installation/`n  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}

$backend = if ($Cpu) { "cpu" } else { $TorchBackend }
$pyExe = ".\.venv\Scripts\python.exe"

Write-Host "Creating .venv (python: $Python) ..."
& uv venv --seed --python $Python .venv

$pipFlags = @()
if ($Force) { $pipFlags += @("--reinstall-package", "torch") }

$target = if ($NoExtras) { "." } else { ".[dev,tensorboard,data]" }
Write-Host "Installing project (editable) from $target with torch backend '$backend' ..."
& uv pip install --python $pyExe --torch-backend=$backend -e $target @pipFlags

Write-Host "`n--- Verification ---"
& $pyExe -c "import torch; print('torch:', torch.__version__); c = torch.cuda.is_available(); print('cuda.is_available:', c); print('device:', torch.cuda.get_device_name(0) if c else 'cpu')"
