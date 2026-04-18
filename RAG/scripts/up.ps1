# Start RAGFlow stack (official docker compose from upstream repo).
$ErrorActionPreference = "Stop"
$RagDir = Split-Path -Parent $PSScriptRoot
$DockerDir = Join-Path $RagDir "ragflow\docker"

if (-not (Test-Path (Join-Path $DockerDir "docker-compose.yml"))) {
    Write-Error "ragflow/docker/docker-compose.yml not found. Run .\scripts\bootstrap.ps1 first."
    exit 1
}

Set-Location $DockerDir
try {
    docker compose -f docker-compose.yml up -d
} finally {
    Set-Location $RagDir
}
