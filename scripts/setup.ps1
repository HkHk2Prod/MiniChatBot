<#
.SYNOPSIS
    Initialize a CUDA-enabled venv for MiniChatBot.

.DESCRIPTION
    Creates .venv (if missing), installs torch from PyTorch's CUDA index,
    then installs the project in editable mode with dev + tensorboard
    extras. Run from the project root.

    CUDA is the default. Pass -Cpu to install the CPU-only wheel.

.PARAMETER Cuda
    CUDA wheel index suffix (cu118, cu121, cu124, cu126). Default: cu124.

.PARAMETER Cpu
    Install CPU-only torch instead of CUDA.

.PARAMETER NoExtras
    Skip the [dev,tensorboard] extras.

.PARAMETER Force
    Force-reinstall torch. Use this when migrating an existing venv from
    CPU torch to CUDA (or vice versa) without re-creating the venv.

.PARAMETER Python
    Python executable used to create the venv. Default: "python".

.EXAMPLE
    .\scripts\setup.ps1                    # default: CUDA cu124, full install
    .\scripts\setup.ps1 -Cuda cu126        # override CUDA version
    .\scripts\setup.ps1 -Cpu               # CPU-only fallback
    .\scripts\setup.ps1 -Force             # swap existing CPU wheel for CUDA
#>

[CmdletBinding()]
param(
    [string]$Cuda = "cu124",
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

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv with $Python ..."
    & $Python -m venv .venv
}

$pyExe = ".\.venv\Scripts\python.exe"
& $pyExe -m pip install --upgrade pip --quiet

$pipFlags = @("--quiet")
if ($Force) { $pipFlags += "--force-reinstall" }

if ($Cpu) {
    Write-Host "Installing torch (CPU) ..."
    & $pyExe -m pip install torch @pipFlags
} else {
    $indexUrl = "https://download.pytorch.org/whl/$Cuda"
    Write-Host "Installing torch from $indexUrl ..."
    & $pyExe -m pip install torch --index-url $indexUrl @pipFlags
}

$target = if ($NoExtras) { "." } else { ".[dev,tensorboard,data]" }
Write-Host "Installing project (editable) from $target ..."
& $pyExe -m pip install -e $target --quiet

Write-Host "`n--- Verification ---"
& $pyExe -c "import torch; print(f'torch: {torch.__version__}'); cuda = torch.cuda.is_available(); print(f'cuda.is_available: {cuda}'); print(f'device: {torch.cuda.get_device_name(0) if cuda else \"cpu\"}')"
