[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$WorktreePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-GitText {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & git @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed"
    }
    return ($output -join [Environment]::NewLine).Trim()
}

try {
    if ([string]::IsNullOrWhiteSpace($WorktreePath)) {
        $root = Invoke-GitText -Arguments @("rev-parse", "--show-toplevel")
    }
    else {
        $candidate = (Resolve-Path -LiteralPath $WorktreePath -ErrorAction Stop).Path
        $root = Invoke-GitText -Arguments @("-C", $candidate, "rev-parse", "--show-toplevel")
    }
    $root = (Resolve-Path -LiteralPath $root -ErrorAction Stop).Path

    $database = Join-Path $root ".codegraph\codegraph.db"
    if (Test-Path -LiteralPath $database -PathType Leaf) {
        Write-Output "CodeGraph is already initialized: $root"
        exit 0
    }

    $codegraph = Get-Command codegraph -ErrorAction SilentlyContinue
    if ($null -eq $codegraph) {
        throw "The 'codegraph' command is not available on PATH"
    }
    $commandPath = $codegraph.Source
    if ([string]::IsNullOrWhiteSpace($commandPath)) {
        $commandPath = $codegraph.Path
    }

    Write-Output "Initializing CodeGraph for: $root"
    & $commandPath init $root
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "codegraph init failed with exit code $exitCode"
    }
    if (-not (Test-Path -LiteralPath $database -PathType Leaf)) {
        throw "codegraph init completed without creating $database"
    }
    Write-Output "CodeGraph initialized: $database"
    exit 0
}
catch {
    $message = if ($null -ne $_ -and $null -ne $_.Exception) {
        $_.Exception.Message
    }
    else {
        "CodeGraph bootstrap failed"
    }
    Write-Error $message
    exit 1
}
