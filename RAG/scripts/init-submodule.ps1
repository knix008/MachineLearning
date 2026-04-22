# Register RAG/ragflow as a git submodule (run from any path; detects repo root).
$ErrorActionPreference = "Stop"

function Get-GitRoot {
    $d = Resolve-Path $PSScriptRoot
    while ($true) {
        $git = Join-Path $d ".git"
        if (Test-Path $git) { return $d }
        $parent = Split-Path $d -Parent
        if ($parent -eq $d) { return $null }
        $d = $parent
    }
}

$Root = Get-GitRoot
if (-not $Root) {
    Write-Error "No .git directory found above scripts folder."
    exit 1
}

$Rel = "RAG/ragflow"
$Full = Join-Path $Root $Rel

if (Test-Path $Full) {
    Write-Error "Path already exists: $Full. Remove it (bootstrap clone) before adding submodule, or keep using bootstrap.ps1 without submodule."
    exit 1
}

Write-Host "[init-submodule] Adding submodule $Rel from repo root: $Root"
Push-Location $Root
try {
    git submodule add https://github.com/infiniflow/ragflow.git $Rel
    git submodule update --init --recursive
} finally {
    Pop-Location
}

Write-Host "[init-submodule] Done."
Write-Host "Edit RAG/.gitignore: remove the 'ragflow/' line so the submodule is not ignored."
Write-Host "Then: git add RAG/.gitignore .gitmodules RAG/ragflow && git commit -m 'Add ragflow submodule'"
