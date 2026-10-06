param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment (.venv) with Python 3.12..."
    py -3.12 -m venv .venv
}

if (-not $SkipInstall) {
    Write-Host "Upgrading pip..."
    & $venvPython -m pip install --upgrade pip
    Write-Host "Installing PyTorch (CUDA 12.4 wheels)..."
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    Write-Host "Installing remaining dependencies..."
    & $venvPython -m pip install -r requirements.txt
}

Write-Host "Starting the detection service on http://127.0.0.1:8000 ..."
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000
