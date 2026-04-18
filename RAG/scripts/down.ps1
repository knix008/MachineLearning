$ErrorActionPreference = "Stop"
$RagDir = Split-Path -Parent $PSScriptRoot
$DockerDir = Join-Path $RagDir "ragflow\docker"

if (-not (Test-Path (Join-Path $DockerDir "docker-compose.yml"))) {
    Write-Error "ragflow/docker not found."
    exit 1
}

Set-Location $DockerDir
try {
    docker compose -f docker-compose.yml down
} finally {
    Set-Location $RagDir
}
