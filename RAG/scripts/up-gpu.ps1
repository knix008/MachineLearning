# Same as up.ps1 but forces DEVICE=gpu for DeepDoc (NVIDIA Container Toolkit required).
$ErrorActionPreference = "Stop"
$RagDir = Split-Path -Parent $PSScriptRoot
$DockerDir = Join-Path $RagDir "ragflow\docker"

if (-not (Test-Path (Join-Path $DockerDir "docker-compose.yml"))) {
    Write-Error "ragflow/docker/docker-compose.yml not found. Run .\scripts\bootstrap.ps1 first."
    exit 1
}

$env:DEVICE = "gpu"
Write-Host "[up-gpu] DEVICE=gpu (host env for docker compose variable expansion)"

Set-Location $DockerDir
try {
    docker compose -f docker-compose.yml up -d
} finally {
    Remove-Item Env:\DEVICE -ErrorAction SilentlyContinue
    Set-Location $RagDir
}
