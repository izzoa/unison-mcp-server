<#
.SYNOPSIS
    Code quality checks script for Unison MCP server on Windows.

.DESCRIPTION
    This PowerShell script performs code quality checks for the Unison MCP server project:
    - Runs static analysis and linting tools on the codebase
    - Ensures code style compliance and detects potential issues
    - Can be integrated into CI/CD pipelines or used locally before commits

.PARAMETER Help
    Displays help information for using the script.

.PARAMETER Verbose
    Enables detailed output during code quality checks.

.EXAMPLE
    .\code_quality_checks.ps1
    Runs all code quality checks on the project.

    .\code_quality_checks.ps1 -Verbose
    Runs code quality checks with detailed output.

.NOTES
    Project Author     : BeehiveInnovations
    Script Author      : GiGiDKR (https://github.com/GiGiDKR)
    Date               : 07-05-2025
    Version            : See project documentation
    References         : https://github.com/izzoa/unison-mcp-server
#>
#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipLinting,
    [switch]$VerboseOutput
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorText {
    param(
        [Parameter(Mandatory)]
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

function Write-Emoji {
    param(
        [Parameter(Mandatory)]
        [string]$Emoji,
        [Parameter(Mandatory)]
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host "$Emoji " -NoNewline
    Write-ColorText $Text -Color $Color
}

Write-Emoji "🔍" "Running Code Quality Checks for Unison MCP Server" -Color Cyan
Write-ColorText "=================================================" -Color Cyan

# Determine Python command
$pythonCmd = $null
$pipCmd = $null

if (Test-Path ".unison_venv") {
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        if (Test-Path ".unison_venv\Scripts\python.exe") {
            $pythonCmd = ".unison_venv\Scripts\python.exe"
            $pipCmd = ".unison_venv\Scripts\pip.exe"
        }
    } else {
        if (Test-Path ".unison_venv/bin/python") {
            $pythonCmd = ".unison_venv/bin/python"
            $pipCmd = ".unison_venv/bin/pip"
        }
    }
    
    if ($pythonCmd) {
        Write-Emoji "✅" "Using venv" -Color Green
    }
} elseif ($env:VIRTUAL_ENV) {
    $pythonCmd = "python"
    $pipCmd = "pip"
    Write-Emoji "✅" "Using activated virtual environment: $env:VIRTUAL_ENV" -Color Green
} else {
    Write-Emoji "❌" "No virtual environment found!" -Color Red
    Write-ColorText "Please run: .\run-server.ps1 first to set up the environment" -Color Yellow
    exit 1
}

Write-Host ""

# Check and install dev dependencies if needed
Write-Emoji "🔍" "Checking development dependencies..." -Color Cyan
$devDepsNeeded = $false

# List of dev tools to check. mypy is included because Step 1b runs it; without
# it here the type-check step would silently skip on a fresh venv.
$devTools = @("ruff", "black", "isort", "mypy", "pytest")

foreach ($tool in $devTools) {
    $toolFound = $false
    
    # Check in venv
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        if (Test-Path ".unison_venv\Scripts\$tool.exe") {
            $toolFound = $true
        }
    } else {
        if (Test-Path ".unison_venv/bin/$tool") {
            $toolFound = $true
        }
    }
    
    # Check in PATH
    if (!$toolFound) {
        try {
            $null = Get-Command $tool -ErrorAction Stop
            $toolFound = $true
        } catch {
            # Tool not found
        }
    }
    
    if (!$toolFound) {
        $devDepsNeeded = $true
        break
    }
}

if ($devDepsNeeded) {
    Write-Emoji "📦" "Installing development dependencies..." -Color Yellow
    try {
        & $pipCmd install -q -r requirements-dev.lock.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install dev dependencies"
        }
        Write-Emoji "✅" "Development dependencies installed" -Color Green
    } catch {
        Write-Emoji "❌" "Failed to install development dependencies" -Color Red
        Write-ColorText "Error: $_" -Color Red
        exit 1
    }
} else {
    Write-Emoji "✅" "Development dependencies already installed" -Color Green
}

# Set tool paths.
#
# Each tool is resolved independently: the presence of one tool in the venv is
# never used to infer another's. The bash gate had exactly that defect — four
# paths gated on a single probe — so a partially-populated venv silently mixed
# venv and system toolchains while still reporting success.
function Resolve-Tool {
    param([string]$Name)

    $venvPath = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        ".unison_venv\Scripts\$Name.exe"
    } else {
        ".unison_venv/bin/$Name"
    }

    if (Test-Path $venvPath) {
        return [pscustomobject]@{ Path = $venvPath; Source = "venv" }
    }
    return [pscustomobject]@{ Path = $Name; Source = "PATH" }
}

$tools = @{}
foreach ($name in @("ruff", "black", "isort", "mypy", "pytest")) {
    $tools[$name] = Resolve-Tool $name
}
$ruffCmd = $tools["ruff"].Path
$blackCmd = $tools["black"].Path
$isortCmd = $tools["isort"].Path
$mypyCmd = $tools["mypy"].Path
$pytestCmd = $tools["pytest"].Path

# Report how each tool resolved, so a mixed toolchain is visible rather than
# silent. A run using a system tool of a different major version can otherwise
# reformat files CI expects in another style and still print success.
$fellBack = @($tools.GetEnumerator() | Where-Object { $_.Value.Source -eq "PATH" })
if ($fellBack.Count -gt 0) {
    Write-Emoji "⚠️" "Some tools resolved outside the venv:" -Color Yellow
    foreach ($entry in ($tools.GetEnumerator() | Sort-Object Key)) {
        $marker = if ($entry.Value.Source -eq "venv") { "venv" } else { "PATH" }
        Write-ColorText "     $($entry.Key.PadRight(8)) -> $marker" -Color Yellow
    }
} else {
    Write-Emoji "✅" "All tools resolved from the venv" -Color Green
}

Write-Host ""

# Step 1: Linting and Formatting
if (!$SkipLinting) {
    Write-Emoji "📋" "Step 1: Running Linting and Formatting Checks" -Color Cyan
    Write-ColorText "--------------------------------------------------" -Color Cyan

    try {
        Write-Emoji "🔧" "Running ruff linting with auto-fix..." -Color Yellow
        & $ruffCmd check --fix --exclude test_simulation_files --exclude .unison_venv
        if ($LASTEXITCODE -ne 0) {
            throw "Ruff linting failed"
        }

        Write-Emoji "🎨" "Running black code formatting..." -Color Yellow
        & $blackCmd . --exclude="test_simulation_files/" --exclude=".unison_venv/"
        if ($LASTEXITCODE -ne 0) {
            throw "Black formatting failed"
        }

        Write-Emoji "📦" "Running import sorting with isort..." -Color Yellow
        & $isortCmd . --skip-glob=".unison_venv/*" --skip-glob="test_simulation_files/*"
        if ($LASTEXITCODE -ne 0) {
            throw "Import sorting failed"
        }

        Write-Emoji "✅" "Verifying all linting passes..." -Color Yellow
        & $ruffCmd check --exclude test_simulation_files --exclude .unison_venv
        if ($LASTEXITCODE -ne 0) {
            throw "Final linting verification failed"
        }

        Write-Emoji "✅" "Step 1 Complete: All linting and formatting checks passed!" -Color Green
    } catch {
        Write-Emoji "❌" "Step 1 Failed: Linting and formatting checks failed" -Color Red
        Write-ColorText "Error: $_" -Color Red
        exit 1
    }
} else {
    Write-Emoji "⏭️" "Skipping linting and formatting checks" -Color Yellow
}

Write-Host ""

# Step 1b: Type Checking (strict allowlist)
#
# The file list mirrors code_quality_checks.sh and the mypy step in
# .github/workflows/test.yml. Keep all three in sync when adding a module to
# the strict allowlist in pyproject.toml.
if (!$SkipLinting) {
    Write-Emoji "🔎" "Step 1b: Running mypy Type Checking" -Color Cyan
    Write-ColorText "---------------------------------------" -Color Cyan

    $mypyAvailable = (Test-Path $mypyCmd) -or (Get-Command $mypyCmd -ErrorAction SilentlyContinue)
    if (!$mypyAvailable) {
        Write-Emoji "⚠️" "mypy not found - skipping type checks (install via: pip install -r requirements-dev.txt)" -Color Yellow
    } else {
        $mypyFiles = @(
            "utils/circuit_breaker.py", "utils/fs_snapshot.py", "utils/tool_execution_context.py", "utils/token_utils.py",
            "providers/shared/provider_type.py", "providers/shared/model_response.py",
            "utils/file_types.py", "utils/security_config.py", "utils/conversation_memory.py",
            "utils/env.py", "utils/model_resolution.py", "utils/request_helpers.py",
            "utils/image_utils.py", "utils/context_reconstructor.py", "utils/file_utils.py",
            "tools/registry.py",
            "scripts/smoke_test_wheel.py", "scripts/build_mockups.py",
            "clink/agents/opencode.py", "clink/parsers/opencode.py",
            "clink/agents/aider.py", "clink/parsers/aider.py",
            "clink/agents/crush.py", "clink/parsers/crush.py",
            "clink/agents/amp.py", "clink/parsers/amp.py",
            "clink/agents/copilot.py", "clink/parsers/copilot.py",
            "utils/observability.py", "utils/json_log_formatter.py"
        )

        Write-Emoji "🔍" "Running mypy on strict allowlist files..." -Color Yellow
        & $mypyCmd @mypyFiles
        if ($LASTEXITCODE -ne 0) {
            Write-Emoji "❌" "Step 1b Failed: Type checking failed" -Color Red
            exit 1
        }
        Write-Emoji "✅" "Step 1b Complete: Type checking passed!" -Color Green
    }
} else {
    Write-Emoji "⏭️" "Skipping type checks" -Color Yellow
}

Write-Host ""

# Step 1c: Mockup drift check
#
# Regenerates the README mockups into a temp dir and compares against the
# checked-in SVGs, catching a scene YAML edit that was never regenerated.
if (!$SkipLinting) {
    Write-Emoji "🔍" "Step 1c: Checking mockup drift" -Color Cyan
    Write-ColorText "----------------------------------" -Color Cyan

    $mockupTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unison-mockups-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $mockupTmp -Force | Out-Null
    try {
        & $pythonCmd scripts/build_mockups.py --output-dir $mockupTmp | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Mockup generation failed"
        }

        $generated = Get-ChildItem -Path $mockupTmp -File | Sort-Object Name
        $committed = Get-ChildItem -Path "docs/assets/mockups" -File | Sort-Object Name
        $differences = @()

        $generatedNames = $generated | ForEach-Object { $_.Name }
        $committedNames = $committed | ForEach-Object { $_.Name }
        $differences += (Compare-Object $generatedNames $committedNames | ForEach-Object { "missing or extra: $($_.InputObject)" })

        foreach ($file in $generated) {
            $counterpart = Join-Path "docs/assets/mockups" $file.Name
            if (Test-Path $counterpart) {
                $a = (Get-FileHash $file.FullName -Algorithm SHA256).Hash
                $b = (Get-FileHash $counterpart -Algorithm SHA256).Hash
                if ($a -ne $b) { $differences += "differs: $($file.Name)" }
            }
        }

        if ($differences.Count -gt 0) {
            Write-Emoji "❌" "Generated SVGs are out of sync with scene YAML." -Color Red
            Write-ColorText "   Run: python scripts/build_mockups.py" -Color Yellow
            $differences | Select-Object -First 20 | ForEach-Object { Write-ColorText "   $_" -Color Yellow }
            exit 1
        }

        Write-Emoji "✅" "Step 1c Complete: Mockups in sync!" -Color Green
    } catch {
        Write-Emoji "❌" "Step 1c Failed: $_" -Color Red
        exit 1
    } finally {
        Remove-Item -Recurse -Force $mockupTmp -ErrorAction SilentlyContinue
    }
} else {
    Write-Emoji "⏭️" "Skipping mockup drift check" -Color Yellow
}

Write-Host ""

# Step 2: Unit Tests
if (!$SkipTests) {
    Write-Emoji "🧪" "Step 2: Running Complete Unit Test Suite" -Color Cyan
    Write-ColorText "---------------------------------------------" -Color Cyan

    try {
        Write-Emoji "🏃" "Running unit tests with coverage (excluding integration tests)..." -Color Yellow

        # Coverage threshold matches code_quality_checks.sh (--cov-fail-under=44).
        # Without it the PowerShell gate could pass a tree the bash gate rejects.
        $pytestArgs = @(
            "tests/", "-v", "-x", "-m", "not integration",
            "--cov=.", "--cov-report=term-missing", "--cov-fail-under=44"
        )
        if ($VerboseOutput) {
            $pytestArgs += "--verbose"
        }

        & $pythonCmd -m pytest @pytestArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Unit tests failed or coverage below threshold"
        }

        Write-Emoji "✅" "Step 2 Complete: All unit tests passed with coverage above threshold!" -Color Green
    } catch {
        Write-Emoji "❌" "Step 2 Failed: Unit tests failed" -Color Red
        Write-ColorText "Error: $_" -Color Red
        exit 1
    }
} else {
    Write-Emoji "⏭️" "Skipping unit tests" -Color Yellow
}

Write-Host ""

# Step 3: Final Summary
Write-Emoji "🎉" "All Code Quality Checks Passed!" -Color Green
Write-ColorText "==================================" -Color Green

if (!$SkipLinting) {
    Write-Emoji "✅" "Linting (ruff): PASSED" -Color Green
    Write-Emoji "✅" "Formatting (black): PASSED" -Color Green
    Write-Emoji "✅" "Import sorting (isort): PASSED" -Color Green
} else {
    Write-Emoji "⏭️" "Linting: SKIPPED" -Color Yellow
}

if (!$SkipTests) {
    Write-Emoji "✅" "Unit tests: PASSED" -Color Green
} else {
    Write-Emoji "⏭️" "Unit tests: SKIPPED" -Color Yellow
}

Write-Host ""
Write-Emoji "🚀" "Your code is ready for commit and GitHub Actions!" -Color Green
Write-Emoji "💡" "Remember to add simulator tests if you modified tools" -Color Yellow
