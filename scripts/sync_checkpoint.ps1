param(
    [string]$Message = "chore: save local research checkpoint"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
    throw "This folder is not connected to Git yet. Follow docs/LOW_MEMORY_EXECUTION_PLAN.md Step 0 first."
}

# Add only project files that exist in this local working tree. Never stage model
# weights, the venv, upstream checkout, raw data or large caches.
$files = @(
    "AGENTS.md", "README.md", "STATUS.md", "LICENSE", ".gitignore",
    "requirements-cpu.txt"
)
foreach ($item in $files) {
    if (Test-Path $item -PathType Leaf) {
        git add -- $item
        if ($LASTEXITCODE -ne 0) { throw "git add failed for $item" }
    }
}
foreach ($folder in @("docs", "configs", "scripts", "tests")) {
    if (Test-Path $folder -PathType Container) {
        git add -- $folder
        if ($LASTEXITCODE -ne 0) { throw "git add failed for $folder" }
    }
}

# Only known small machine-readable reports are eligible for version control.
$reportPaths = @(
    "results/environment.json",
    "results/dataset_manifest.json",
    "results/ci_run_manifest.json",
    "results/local_environment_2026-10-26.json"
)
foreach ($relative in $reportPaths) {
    if (Test-Path $relative -PathType Leaf) {
        git add -- $relative
        if ($LASTEXITCODE -ne 0) { throw "git add failed for $relative" }
    }
}
foreach ($folder in @("results/smoke", "results/pilot")) {
    if (Test-Path $folder -PathType Container) {
        Get-ChildItem $folder -File -Filter "*.json" | ForEach-Object {
            $relative = Resolve-Path $_.FullName -Relative
            git add -- $relative
            if ($LASTEXITCODE -ne 0) { throw "git add failed for $relative" }
        }
    }
}

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No staged project/report changes; nothing to save."
    exit 0
}
if ($LASTEXITCODE -ne 1) {
    throw "Unable to inspect staged changes."
}

Write-Host "Reviewing staged files before commit:"
git diff --cached --stat
git diff --cached --name-only
Write-Host "Confirm no model, raw data, virtual environment, or upstream checkout is staged before continuing."
$confirmation = Read-Host "Type COMMIT to commit and push these staged changes"
if ($confirmation -cne "COMMIT") {
    Write-Host "Cancelled; no commit or push performed."
    exit 1
}

git commit -m $Message
if ($LASTEXITCODE -ne 0) { throw "Commit failed. Review git status." }

git push
if ($LASTEXITCODE -ne 0) {
    throw "Commit is local but push failed. Resolve the remote issue and retry git push."
}

Write-Host "Saved and pushed checkpoint successfully."
git log -1 --oneline
