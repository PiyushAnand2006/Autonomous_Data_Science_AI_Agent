# AADS — Clone Reference Repositories
# Usage: .\aads\scripts\clone_references.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$RefDir = Join-Path $RepoRoot "references"

if (-not (Test-Path $RefDir)) {
    New-Item -ItemType Directory -Path $RefDir -Force | Out-Null
}

$repos = @(
    @{ Name = "ai-data-science-team"; Url = "https://github.com/business-science/ai-data-science-team.git" },
    @{ Name = "DeepAnalyze";          Url = "https://github.com/ruc-datalab/DeepAnalyze.git" },
    @{ Name = "DatawiseAgent";        Url = "https://github.com/zimingyou01/DatawiseAgent.git" }
)

foreach ($repo in $repos) {
    $dest = Join-Path $RefDir $repo.Name
    if (Test-Path $dest) {
        Write-Host "[SKIP] $($repo.Name) already exists at $dest"
    } else {
        Write-Host "[CLONE] $($repo.Name) -> $dest"
        git clone $repo.Url $dest
    }
}

Write-Host ""
Write-Host "Done. Reference repositories are in: $RefDir"
Write-Host "REMINDER: These are READ-ONLY. Do not modify files inside references/."
