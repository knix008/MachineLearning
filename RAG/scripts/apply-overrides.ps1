# Merge RAG/config/env.override into ragflow/docker/.env (KEY=VAL lines).
# Backup: ragflow/docker/.env.bak
$ErrorActionPreference = "Stop"
$RagDir = Split-Path -Parent $PSScriptRoot
$OverridePath = Join-Path $RagDir "config\env.override"
$DockerDir = Join-Path $RagDir "ragflow\docker"
$EnvPath = Join-Path $DockerDir ".env"

if (-not (Test-Path $OverridePath)) {
    Write-Host "[apply-overrides] No config\env.override — copy from config\env.override.example"
    exit 0
}
if (-not (Test-Path $EnvPath)) {
    Write-Error "Missing $EnvPath — run bootstrap.ps1 first."
    exit 1
}

$overrides = @{}
Get-Content $OverridePath | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $eq = $line.IndexOf("=")
    if ($eq -lt 1) { return }
    $key = $line.Substring(0, $eq).Trim()
    $val = $line.Substring($eq + 1).Trim()
    if ($key) { $overrides[$key] = $val }
}
if ($overrides.Count -eq 0) {
    Write-Host "[apply-overrides] No key=value entries in env.override"
    exit 0
}

$lines = @(Get-Content $EnvPath)
$seen = @{}
$out = [System.Collections.Generic.List[string]]::new()
foreach ($ln in $lines) {
    $trim = $ln.Trim()
    if ($trim -match "^([A-Za-z_][A-Za-z0-9_]*)=") {
        $k = $Matches[1]
        if ($overrides.ContainsKey($k)) {
            [void]$seen.Add($k, $true)
            $out.Add("$k=$($overrides[$k])")
            continue
        }
    }
    $out.Add($ln)
}
foreach ($k in $overrides.Keys) {
    if (-not $seen.ContainsKey($k)) {
        $out.Add("$k=$($overrides[$k])")
    }
}

Copy-Item $EnvPath "$EnvPath.bak" -Force
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines($EnvPath, $out.ToArray(), $utf8NoBom)
Write-Host "[apply-overrides] Updated $EnvPath ($($overrides.Count) keys). Backup: .env.bak"
