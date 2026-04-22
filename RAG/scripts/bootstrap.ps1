# Clone infiniflow/ragflow into RAG/ragflow (gitignored unless using submodule).
param(
    # Optional: e.g. v0.24.0 — run after clone (or on existing repo) to match image/entrypoint
    [string] $Tag = ""
)

$ErrorActionPreference = "Stop"
$RagDir = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $RagDir "ragflow"

function Invoke-CheckoutTag {
    param([string]$Path, [string]$Ref)
    if (-not $Ref) { return }
    Write-Host "[bootstrap] Checking out $Ref in $Path"
    Push-Location $Path
    try {
        git fetch --tags origin 2>$null
        git checkout $Ref
    } finally {
        Pop-Location
    }
}

if (Test-Path $Target) {
    Write-Host "[bootstrap] Already exists: $Target"
    if ($Tag) { Invoke-CheckoutTag -Path $Target -Ref $Tag }
    exit 0
}

Write-Host "[bootstrap] Cloning RAGFlow -> $Target"
git clone https://github.com/infiniflow/ragflow.git $Target
if ($Tag) { Invoke-CheckoutTag -Path $Target -Ref $Tag }
Write-Host "[bootstrap] Done. Optional: copy config\env.override.example to config\env.override then .\scripts\apply-overrides.ps1"
Write-Host "[bootstrap] Start: .\scripts\up.ps1 or .\scripts\up-gpu.ps1"
