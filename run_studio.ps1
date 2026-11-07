$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not active. Activate your project virtual environment first."
    exit 1
}
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
