[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $rootOutput = & git rev-parse --show-toplevel 2>$null
    $root = if ($null -eq $rootOutput) { "" } else { ($rootOutput -join [Environment]::NewLine).Trim() }
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($root)) {
        throw "Run this script from a Git worktree"
    }
    $root = (Resolve-Path -LiteralPath $root -ErrorAction Stop).Path

    $hook = Join-Path $root ".githooks\post-checkout"
    $bootstrap = Join-Path $root "scripts\ensure-codegraph.ps1"
    if (-not (Test-Path -LiteralPath $hook -PathType Leaf)) {
        throw "Missing tracked hook: $hook"
    }
    if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
        throw "Missing bootstrap script: $bootstrap"
    }

    $configuredOutput = & git config --get core.hooksPath 2>$null
    $configured = if ($null -eq $configuredOutput) { "" } else { ($configuredOutput -join [Environment]::NewLine).Trim() }
    if (-not [string]::IsNullOrWhiteSpace($configured)) {
        $configuredPath = if ([IO.Path]::IsPathRooted($configured)) {
            [IO.Path]::GetFullPath($configured)
        }
        else {
            [IO.Path]::GetFullPath((Join-Path $root $configured))
        }
        $expectedPath = [IO.Path]::GetFullPath((Join-Path $root ".githooks"))
        if ($configuredPath -ne $expectedPath) {
            throw "core.hooksPath is already configured as '$configured'; refusing to replace it"
        }
    }

    $hooksDirectory = Split-Path -Parent $hook
    & git config core.hooksPath $hooksDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to configure core.hooksPath"
    }
    Write-Output "Configured core.hooksPath for this repository: $hooksDirectory"

    & $bootstrap -WorktreePath $root
    if ($LASTEXITCODE -ne 0) {
        throw "The current worktree could not be initialized"
    }
}
catch {
    $message = if ($null -ne $_ -and $null -ne $_.Exception) {
        $_.Exception.Message
    }
    else {
        "CodeGraph hook installation failed"
    }
    Write-Error $message
    exit 1
}
