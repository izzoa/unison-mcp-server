<#
.SYNOPSIS
    Installation, configuration, and launch script for Unison MCP server on Windows.

.DESCRIPTION
    This PowerShell script prepares the environment for the Unison MCP server:
    - Installs and checks Python 3.10+ (with venv or uv if available)
    - Installs required Python dependencies
    - Configures environment files (.env)
    - Validates presence of required API keys
    - Cleans Python caches and obsolete Docker artifacts
    - Offers automatic integration with Claude Desktop, Gemini CLI, VSCode, Cursor, Windsurf, and Trae
    - Manages configuration file backups (max 3 retained)
    - Allows real-time log following or server launch

.PARAMETER Help
    Shows script help.

.PARAMETER Version
    Shows Unison MCP server version.

.PARAMETER Follow
    Follows server logs in real time.

.PARAMETER Config
    Shows configuration instructions for Claude and other compatible clients.

.PARAMETER ClearCache
    Removes Python cache files (__pycache__, .pyc).

.PARAMETER SkipVenv
    Skips Python virtual environment creation.

.PARAMETER SkipDocker
    Skips Docker checks and cleanup.

.PARAMETER Force
    Forces recreation of the Python virtual environment.
    
.PARAMETER VerboseOutput
    Enables more detailed output (currently unused).

.PARAMETER Dev
    Installs development dependencies from requirements-dev.txt if available.

.PARAMETER Docker
    Uses Docker to build and run the MCP server instead of Python virtual environment.

.EXAMPLE
    .\run-server.ps1
    Prepares the environment and starts the Unison MCP server.

    .\run-server.ps1 -Follow
    Follows server logs in real time.

    .\run-server.ps1 -Config
    Shows configuration instructions for clients.

    .\run-server.ps1 -Dev
    Prepares the environment with development dependencies and starts the server.

    .\run-server.ps1 -Docker
    Builds and runs the server using Docker containers.

    .\run-server.ps1 -Docker -Follow
    Builds and runs the server using Docker containers and follows the logs.

    .\run-server.ps1 -Docker -Force
    Forces rebuilding of the Docker image and runs the server.

.NOTES
    Project Author     : BeehiveInnovations
    Script Author      : GiGiDKR (https://github.com/GiGiDKR)
    Date               : 07-05-2025
    Version            : See config.py (__version__)
    References         : https://github.com/izzoa/unison-mcp-server

    REQUIREMENTS
    PowerShell 7.0 or later. The Windows PowerShell 5.1 that ships with
    Windows 10/11 cannot parse this script. Install with:
        winget install --id Microsoft.PowerShell --source winget

    WSL is NOT required. This script is native PowerShell and invokes no
    bash, no WSL, and no Unix utilities.

    EXECUTION POLICY
    Windows blocks unsigned scripts by default, so this script may fail
    before its first line with "running scripts is disabled on this system".
    Allow it for the current process only:
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

    A process-scoped policy expires with the shell and needs no administrator
    rights, but it CANNOT override a MachinePolicy or UserPolicy set by Group
    Policy — those take precedence. On a managed machine, contact your
    administrator.

#>
# PowerShell 7.0+ is required: this script uses the ternary operator, which
# Windows PowerShell 5.1 cannot parse. PowerShell parses a script in full
# before executing any of it, so declaring 5.1 here produced an unexplained
# parse error on the interpreter shipped with Windows rather than a version
# message. Raise this deliberately if later syntax is adopted.
#Requires -Version 7.0
[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$Version,
    [switch]$Follow,
    [switch]$Config,
    [switch]$ClearCache,
    [switch]$SkipVenv,
    [switch]$SkipDocker,
    [switch]$Force,
    [switch]$VerboseOutput,
    [switch]$Dev,
    [switch]$Docker
)

# ============================================================================
# Unison MCP Server Setup Script for Windows
# 
# A Windows-compatible setup script that handles environment setup, 
# dependency installation, and configuration.
# ============================================================================

# Set error action preference
$ErrorActionPreference = "Stop"

# uv validates TLS against bundled roots by default, which fails behind
# corporate TLS-intercepting proxies whose CA lives in the OS store (pip
# already trusts the OS store). Opt uv into the system trust store for every
# uv invocation this script makes. UV_NATIVE_TLS is the deprecated alias,
# kept for older uv versions.
$env:UV_SYSTEM_CERTS = "true"
$env:UV_NATIVE_TLS = "1"

# uv installs by hardlinking from its cache, which OneDrive-synced folders
# reject ("incompatible hardlinks", os error 396) — corporate checkouts often
# live under OneDrive. Copying is slightly slower but always works.
$env:UV_LINK_MODE = "copy"

# ----------------------------------------------------------------------------
# Constants and Configuration  
# ----------------------------------------------------------------------------

$script:VENV_PATH = ".unison_venv"
$script:DOCKER_CLEANED_FLAG = ".docker_cleaned"
$script:DESKTOP_CONFIG_FLAG = ".desktop_configured"
$script:LOG_DIR = "logs"
$script:LOG_FILE = "mcp_server.log"
$script:LegacyServerNames = @("zen", "zen-mcp", "zen-mcp-server", "zen_mcp", "zen_mcp_server", "pal", "pal-mcp", "pal-mcp-server", "pal_mcp", "pal_mcp_server")

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

function Write-Success {
    param([string]$Message)
    Write-Host "✓ " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Write text as UTF-8 guaranteed WITHOUT a byte-order mark, regardless of
# PowerShell version or $PSDefaultParameterValues encoding overrides (Windows
# PowerShell 5.1's `-Encoding UTF8` prepends a BOM; pwsh 7 defaults can be
# reconfigured). Node-based MCP hosts (Claude Desktop, Gemini CLI, Qwen CLI,
# Cursor, ...) load their JSON configs with JSON.parse, which rejects a
# leading BOM — the whole config is then silently ignored and the server
# never appears in the host.
function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Content
    )
    $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    [System.IO.File]::WriteAllText($resolved, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

# Alternative method to force remove locked directories
function Remove-LockedDirectory {
    param([string]$Path)
    
    if (!(Test-Path $Path)) {
        return $true
    }
    
    try {
        # Try standard removal first
        Remove-Item -Recurse -Force $Path -ErrorAction Stop
        return $true
    }
    catch {
        Write-Warning "Standard removal failed, trying alternative methods..."
        
        # Method 1: Use takeown and icacls to force ownership
        try {
            Write-Info "Attempting to take ownership of locked files..."
            takeown /F "$Path" /R /D Y 2>$null | Out-Null
            icacls "$Path" /grant administrators:F /T 2>$null | Out-Null
            Remove-Item -Recurse -Force $Path -ErrorAction Stop
            return $true
        }
        catch {
            Write-Warning "Ownership method failed"
        }
        
        # Method 2: Rename and schedule for deletion on reboot
        try {
            $tempName = "$Path.delete_$(Get-Random)"
            Write-Info "Renaming to: $tempName (will be deleted on next reboot)"
            Rename-Item $Path $tempName -ErrorAction Stop
            
            # Schedule for deletion on reboot using movefile
            if (Get-Command "schtasks" -ErrorAction SilentlyContinue) {
                Write-Info "Scheduling for deletion on next reboot..."
            }
            
            Write-Warning "Environment renamed to $tempName and will be deleted on next reboot"
            return $true
        }
        catch {
            Write-Warning "Rename method failed"
        }
        
        # If all methods fail, return false
        return $false
    }
}

# Remove legacy MCP server entries from a hash/dictionary or PSObject
function Remove-LegacyServerKeys {
    param([object]$Container)

    $removed = $false
    if ($null -eq $Container) {
        return $false
    }

    foreach ($legacy in $script:LegacyServerNames) {
        if ($Container -is [System.Collections.IDictionary]) {
            if ($Container.Contains($legacy)) {
                $Container.Remove($legacy) | Out-Null
                $removed = $true
            }
        }
        elseif ($Container.PSObject -and $Container.PSObject.Properties[$legacy]) {
            $Container.PSObject.Properties.Remove($legacy)
            $removed = $true
        }
    }

    return $removed
}

# Manage configuration file backups with maximum 3 files retention
function Manage-ConfigBackups {
    param(
        [string]$ConfigFilePath,
        [int]$MaxBackups = 3
    )
    
    if (!(Test-Path $ConfigFilePath)) {
        Write-Warning "Configuration file not found: $ConfigFilePath"
        return $null
    }
    
    try {
        # Create new backup with timestamp
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $backupPath = "$ConfigFilePath.backup_$timestamp"
        Copy-Item $ConfigFilePath $backupPath -ErrorAction Stop
        
        # Find all existing backups for this config file
        $configDir = Split-Path $ConfigFilePath -Parent
        $configFileName = Split-Path $ConfigFilePath -Leaf
        $backupPattern = "$configFileName.backup_*"
        
        $existingBackups = Get-ChildItem -Path $configDir -Filter $backupPattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
        
        # Keep only the most recent MaxBackups files
        if ($existingBackups.Count -gt $MaxBackups) {
            $backupsToRemove = $existingBackups | Select-Object -Skip $MaxBackups
            foreach ($backup in $backupsToRemove) {
                try {
                    Remove-Item $backup.FullName -Force -ErrorAction Stop
                    Write-Info "Removed old backup: $($backup.Name)"
                }
                catch {
                    Write-Warning "Could not remove old backup: $($backup.Name)"
                }
            }
            Write-Success "Backup retention: kept $MaxBackups most recent backups"
        }
        
        Write-Success "Backup created: $(Split-Path $backupPath -Leaf)"
        return $backupPath
        
    }
    catch {
        Write-Warning "Failed to create backup: $_"
        return $null
    }
}

# Get version from config.py
function Get-Version {
    try {
        if (Test-Path "config.py") {
            $content = Get-Content "config.py" -ErrorAction Stop
            $versionLine = $content | Where-Object { $_ -match '^__version__ = ' }
            if ($versionLine) {
                return ($versionLine -replace '__version__ = "([^"]*)"', '$1')
            }
        }
        return "unknown"
    }
    catch {
        return "unknown"
    }
}

# Clear Python cache files
function Clear-PythonCache {
    Write-Info "Clearing Python cache files..."
    
    try {
        # Remove .pyc files
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Remove __pycache__ directories
        Get-ChildItem -Path . -Recurse -Name "__pycache__" -Directory -ErrorAction SilentlyContinue | 
        ForEach-Object { Remove-Item -Path $_ -Recurse -Force }
        
        Write-Success "Python cache cleared"
    }
    catch {
        Write-Warning "Could not clear all cache files: $_"
    }
}

# Get absolute path
function Get-AbsolutePath {
    param([string]$Path)
    
    if (Test-Path $Path) {
        # Use Resolve-Path for full resolution
        return Resolve-Path $Path
    }
    else {
        # Use unresolved method
        return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    }
}

# Check Python version
function Test-PythonVersion {
    param([string]$PythonCmd)
    try {
        $version = & $PythonCmd --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 10)
        }
        return $false
    }
    catch {
        return $false
    }
}

# Find Python installation
function Find-Python {
    $pythonCandidates = @("python", "python3", "py")
    
    foreach ($cmd in $pythonCandidates) {
        if (Test-Command $cmd) {
            if (Test-PythonVersion $cmd) {
                $version = & $cmd --version 2>&1
                Write-Success "Found Python: $version"
                return $cmd
            }
        }
    }
    
    # Try Windows Python Launcher with specific versions
    $pythonVersions = @("3.12", "3.11", "3.10", "3.9")
    foreach ($version in $pythonVersions) {
        $cmd = "py -$version"
        try {
            $null = Invoke-Expression "$cmd --version" 2>$null
            Write-Success "Found Python via py launcher: $cmd"
            return $cmd
        }
        catch {
            continue
        }
    }
    
    Write-Error "Python 3.10+ not found. Please install Python from https://python.org"
    return $null
}

# Clean up old Docker artifacts
function Cleanup-Docker {
    if (Test-Path $DOCKER_CLEANED_FLAG) {
        return
    }
    
    if (!(Test-Command "docker")) {
        return
    }
    
    try {
        $null = docker info 2>$null
    }
    catch {
        return
    }
    
    $foundArtifacts = $false
    
    # Define containers to remove
    $containers = @(
        "gemini-mcp-server",
        "gemini-mcp-redis", 
        "unison-mcp-server",
        "unison-mcp-redis",
        "unison-mcp-log-monitor"
    )
    
    # Remove containers
    foreach ($container in $containers) {
        try {
            $exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $container }
            if ($exists) {
                if (!$foundArtifacts) {
                    Write-Info "One-time Docker cleanup..."
                    $foundArtifacts = $true
                }
                Write-Info "  Removing container: $container"
                docker stop $container 2>$null | Out-Null
                docker rm $container 2>$null | Out-Null
            }
        }
        catch {
            # Ignore errors
        }
    }
    
    # Remove images
    $images = @("gemini-mcp-server:latest", "unison-mcp-server:latest")
    foreach ($image in $images) {
        try {
            $exists = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -eq $image }
            if ($exists) {
                if (!$foundArtifacts) {
                    Write-Info "One-time Docker cleanup..."
                    $foundArtifacts = $true
                }
                Write-Info "  Removing image: $image"
                docker rmi $image 2>$null | Out-Null
            }
        }
        catch {
            # Ignore errors
        }
    }
    
    # Remove volumes
    $volumes = @("redis_data", "mcp_logs")
    foreach ($volume in $volumes) {
        try {
            $exists = docker volume ls --format "{{.Name}}" | Where-Object { $_ -eq $volume }
            if ($exists) {
                if (!$foundArtifacts) {
                    Write-Info "One-time Docker cleanup..."
                    $foundArtifacts = $true
                }
                Write-Info "  Removing volume: $volume"
                docker volume rm $volume 2>$null | Out-Null
            }
        }
        catch {
            # Ignore errors
        }
    }
    
    if ($foundArtifacts) {
        Write-Success "Docker cleanup complete"
    }
    
    New-Item -Path $DOCKER_CLEANED_FLAG -ItemType File -Force | Out-Null
}

# True when an .env value is present and is not the template placeholder for
# the given variable name (placeholders look like your_<name-lowercase>_here)
function Test-ApiKeyValueConfigured {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    if ($Value -eq "your_$($Name.ToLower())_here") { return $false }
    return $true
}

# Validate API keys
function Test-ApiKeys {
    Write-Step "Validating API Keys"
    
    if (!(Test-Path ".env")) {
        Write-Warning "No .env file found. API keys should be configured."
        return $false
    }
    
    $envContent = Get-Content ".env"
    $hasValidKey = $false
    
    # Native API-key providers recognized in .env. A value equal to its
    # template placeholder is unconfigured. Provider key-format regexes are
    # deliberately NOT enforced here: formats change (e.g. current OpenAI
    # sk-proj-... keys) and a stale pattern would reject real credentials.
    $apiKeyNames = @(
        "GEMINI_API_KEY"
        "OPENAI_API_KEY"
        "ANTHROPIC_API_KEY"
        "XAI_API_KEY"
        "OPENROUTER_API_KEY"
        "DIAL_API_KEY"
    )
    
    $envValues = @{}
    foreach ($line in $envContent) {
        if ($line -match '^([^#][^=]*?)=(.*)$') {
            $envValues[$matches[1].Trim()] = $matches[2].Trim() -replace '^["'']|["'']$', ''
        }
    }

    foreach ($name in $apiKeyNames) {
        if ($envValues.ContainsKey($name) -and (Test-ApiKeyValueConfigured -Name $name -Value $envValues[$name])) {
            Write-Success "Found valid $name"
            $hasValidKey = $true
        }
    }

    # Azure OpenAI is configured as a key + endpoint pair
    if ((Test-ApiKeyValueConfigured -Name "AZURE_OPENAI_API_KEY" -Value $envValues["AZURE_OPENAI_API_KEY"]) -and
        (Test-ApiKeyValueConfigured -Name "AZURE_OPENAI_ENDPOINT" -Value $envValues["AZURE_OPENAI_ENDPOINT"])) {
        Write-Success "Found Azure OpenAI configuration"
        $hasValidKey = $true
    }

    # Custom endpoints (Ollama, vLLM, ...) may be keyless: a real
    # CUSTOM_API_URL alone is a configured provider
    if (Test-ApiKeyValueConfigured -Name "CUSTOM_API_URL" -Value $envValues["CUSTOM_API_URL"]) {
        Write-Success "Found custom API endpoint (CUSTOM_API_URL)"
        $hasValidKey = $true
    }
    
    if (!$hasValidKey) {
        Write-Warning "No valid API keys found in .env file"
        Write-Info "Please edit .env file with your actual API keys"
        return $false
    }
    
    return $true
}

# Check if uv is available
function Test-Uv {
    return Test-Command "uv"
}

# Setup environment using uv-first approach
function Initialize-Environment {
    Write-Step "Setting up Python Environment"
    
    # Try uv first for faster package management
    if (Test-Uv) {
        Write-Info "Using uv for faster package management..."
        
        if (Test-Path $VENV_PATH) {
            if ($Force) {
                Write-Warning "Removing existing environment..."
                Remove-Item -Recurse -Force $VENV_PATH
            }
            else {
                Write-Success "Virtual environment already exists"
                $pythonPath = "$VENV_PATH\Scripts\python.exe"
                if (Test-Path $pythonPath) {
                    return Get-AbsolutePath $pythonPath
                }
            }
        }
        
        try {
            Write-Info "Creating virtual environment with uv..."
            # --seed installs pip into the venv: uv-created venvs are pip-less
            # by default, which would strand the pip fallback path
            uv venv --seed $VENV_PATH --python 3.12
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Environment created with uv"
                return Get-AbsolutePath "$VENV_PATH\Scripts\python.exe"
            }
            Write-Warning "uv could not create the environment (exit code $LASTEXITCODE) - falling back to a system Python venv"
        }
        catch {
            Write-Warning "uv failed, falling back to venv"
        }
    }
    
    # Fallback to standard venv
    $pythonCmd = Find-Python
    if (!$pythonCmd) {
        throw "Python 3.10+ not found"
    }
    
    if (Test-Path $VENV_PATH) {
        if ($Force) {
            Write-Warning "Removing existing environment..."
            try {
                # Stop any Python processes that might be using the venv
                Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$VENV_PATH*" } | Stop-Process -Force -ErrorAction SilentlyContinue
                
                # Wait a moment for processes to terminate
                Start-Sleep -Seconds 2
                
                # Use the robust removal function
                if (Remove-LockedDirectory $VENV_PATH) {
                    Write-Success "Existing environment removed"
                }
                else {
                    throw "Unable to remove existing environment. Please restart your computer and try again."
                }
                
            }
            catch {
                Write-Error "Failed to remove existing environment: $_"
                Write-Host ""
                Write-Host "Try these solutions:" -ForegroundColor Yellow
                Write-Host "1. Close all terminals and VS Code instances" -ForegroundColor White
                Write-Host "2. Run: Get-Process python* | Stop-Process -Force" -ForegroundColor White
                Write-Host "3. Manually delete: $VENV_PATH" -ForegroundColor White
                Write-Host "4. Then run the script again" -ForegroundColor White
                exit 1
            }
        }
        else {
            Write-Success "Virtual environment already exists"
            return Get-AbsolutePath "$VENV_PATH\Scripts\python.exe"
        }
    }
    
    Write-Info "Creating virtual environment with $pythonCmd..."
    if ($pythonCmd.StartsWith("py ")) {
        Invoke-Expression "$pythonCmd -m venv $VENV_PATH"
    }
    else {
        & $pythonCmd -m venv $VENV_PATH
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    
    Write-Success "Virtual environment created"
    return Get-AbsolutePath "$VENV_PATH\Scripts\python.exe"
}

# Setup virtual environment (legacy function for compatibility)
function Initialize-VirtualEnvironment {
    Write-Step "Setting up Python Virtual Environment"
    
    if (!$SkipVenv -and (Test-Path $VENV_PATH)) {
        if ($Force) {
            Write-Warning "Removing existing virtual environment..."
            try {
                # Stop any Python processes that might be using the venv
                Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$VENV_PATH*" } | Stop-Process -Force -ErrorAction SilentlyContinue
                
                # Wait a moment for processes to terminate
                Start-Sleep -Seconds 2
                
                # Use the robust removal function
                if (Remove-LockedDirectory $VENV_PATH) {
                    Write-Success "Existing environment removed"
                }
                else {
                    throw "Unable to remove existing environment. Please restart your computer and try again."
                }
                
            }
            catch {
                Write-Error "Failed to remove existing environment: $_"
                Write-Host ""
                Write-Host "Try these solutions:" -ForegroundColor Yellow
                Write-Host "1. Close all terminals and VS Code instances" -ForegroundColor White
                Write-Host "2. Run: Get-Process python* | Stop-Process -Force" -ForegroundColor White
                Write-Host "3. Manually delete: $VENV_PATH" -ForegroundColor White
                Write-Host "4. Then run the script again" -ForegroundColor White
                exit 1
            }
        }
        else {
            Write-Success "Virtual environment already exists"
            return
        }
    }
    
    if ($SkipVenv) {
        Write-Warning "Skipping virtual environment setup"
        return
    }
    
    $pythonCmd = Find-Python
    if (!$pythonCmd) {
        Write-Error "Python 3.10+ not found. Please install Python from https://python.org"
        exit 1
    }
    
    Write-Info "Using Python: $pythonCmd"
    Write-Info "Creating virtual environment..."
    
    try {
        if ($pythonCmd.StartsWith("py ")) {
            Invoke-Expression "$pythonCmd -m venv $VENV_PATH"
        }
        else {
            & $pythonCmd -m venv $VENV_PATH
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        
        Write-Success "Virtual environment created"
    }
    catch {
        Write-Error "Failed to create virtual environment: $_"
        exit 1
    }
}

# Install dependencies function - Simplified uv-first approach
function Install-Dependencies {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [switch]$InstallDevDependencies = $false
    )
    
    Write-Step "Installing Dependencies"

    # Build requirements files list
    $requirementsFiles = @("requirements.lock.txt")
    if ($InstallDevDependencies) {
        if (Test-Path "requirements-dev.lock.txt") {
            $requirementsFiles += "requirements-dev.lock.txt"
            Write-Info "Including development dependencies from requirements-dev.lock.txt"
        }
        else {
            Write-Warning "Development dependencies requested but requirements-dev.lock.txt not found"
        }
    }

    # Try uv first for faster package management
    $useUv = Test-Uv
    if ($useUv) {
        Write-Info "Installing dependencies with uv (fast)..."
        try {
            foreach ($file in $requirementsFiles) {
                Write-Info "Installing from $file with uv..."
                $uv = (Get-Command uv -ErrorAction Stop).Source
                # Direct invocation passes each argument intact; Start-Process
                # -ArgumentList joins elements unquoted, splitting paths that
                # contain spaces (e.g. "OneDrive - Vendor") into fragments.
                & $uv pip install -r $file --python $PythonPath
                if ($LASTEXITCODE -ne 0) {
                    throw "uv failed to install $file with exit code $LASTEXITCODE"
                }
            }
            Write-Success "Dependencies installed successfully with uv"
            return
        }
        catch {
            Write-Warning "uv installation failed: $_. Falling back to pip"
            $useUv = $false
        }
    }

    # Fallback to pip
    Write-Info "Installing dependencies with pip..."
    $pipCmd = Join-Path (Split-Path $PythonPath -Parent) "pip.exe"
    if (!(Test-Path $pipCmd)) {
        # A venv created by uv without --seed has no pip at all; bootstrap it
        # so the fallback can proceed.
        Write-Info "pip not present in venv - bootstrapping via ensurepip..."
        & $PythonPath -m ensurepip --upgrade *> $null
        if (!(Test-Path $pipCmd)) {
            Write-Error "pip is unavailable in the virtual environment and ensurepip could not install it."
            exit 1
        }
    }
    
    try {
        # Upgrade pip via the interpreter; pip.exe cannot modify itself on
        # Windows and always errors when asked to
        & $PythonPath -m pip install --upgrade pip *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Could not upgrade pip, continuing..."
        }
    }
    catch {
        Write-Warning "Could not upgrade pip, continuing..."
    }

    try {
        foreach ($file in $requirementsFiles) {
            Write-Info "Installing from $file with pip..."
            & $pipCmd install -r $file
            if ($LASTEXITCODE -ne 0) {
                throw "pip failed to install $file"
            }
        }
        Write-Success "Dependencies installed successfully with pip"
    }
    catch {
        Write-Error "Failed to install dependencies with pip: $_"
        exit 1
    }
}

# ----------------------------------------------------------------------------
# Docker Functions
# ============================================================================

# Test Docker availability and requirements
function Test-DockerRequirements {
    Write-Step "Checking Docker Requirements"
    
    if (!(Test-Command "docker")) {
        Write-Error "Docker not found. Please install Docker Desktop from https://docker.com"
        return $false
    }
    
    try {
        $null = docker version 2>$null
        Write-Success "Docker is installed and running"
    }
    catch {
        Write-Error "Docker is installed but not running. Please start Docker Desktop."
        return $false
    }
    
    if (!(Test-Command "docker-compose")) {
        Write-Warning "docker-compose not found. Trying docker compose..."
        try {
            $null = docker compose version 2>$null
            Write-Success "Docker Compose (v2) is available"
            return $true
        }
        catch {
            Write-Error "Docker Compose not found. Please install Docker Compose."
            return $false
        }
    }
    else {
        Write-Success "Docker Compose is available"
        return $true
    }
}

# Build Docker image
function Build-DockerImage {
    param([switch]$Force = $false)
    
    Write-Step "Building Docker Image"
    
    # Check if image exists
    try {
        $imageExists = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -eq "unison-mcp-server:latest" }
        if ($imageExists -and !$Force) {
            Write-Success "Docker image already exists. Use -Force to rebuild."
            return $true
        }
    }
    catch {
        # Continue if command fails
    }
    
    if ($Force -and $imageExists) {
        Write-Info "Forcing rebuild of Docker image..."
        try {
            docker rmi unison-mcp-server:latest 2>$null
        }
        catch {
            Write-Warning "Could not remove existing image, continuing..."
        }
    }
    
    Write-Info "Building Docker image from Dockerfile..."
    try {
        $buildArgs = @()
        if ($Dev) {
            # For development builds, we could add specific build args
            Write-Info "Building with development support..."
        }
        
        docker build -t unison-mcp-server:latest .
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed"
        }
        
        Write-Success "Docker image built successfully"
        return $true
    }
    catch {
        Write-Error "Failed to build Docker image: $_"
        return $false
    }
}

# Prepare Docker environment file
function Initialize-DockerEnvironment {
    Write-Step "Preparing Docker Environment"
    
    # Ensure .env file exists
    if (!(Test-Path ".env")) {
        Write-Warning "No .env file found. Creating default .env file..."
        
        $defaultEnv = @"
# API keys — the server enables one provider per REAL value below. ONLY these
# variables activate providers; placeholder values count as unset.
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
DIAL_API_KEY=your_dial_api_key_here

# Azure OpenAI (both required together)
#AZURE_OPENAI_API_KEY=
#AZURE_OPENAI_ENDPOINT=

# Local/self-hosted OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, ...).
# A real URL alone is enough; leave CUSTOM_API_KEY empty for keyless servers.
#CUSTOM_API_URL=http://localhost:11434/v1
#CUSTOM_API_KEY=
#CUSTOM_MODEL_NAME=llama3.2

# DIAL extras (only meaningful alongside a real DIAL_API_KEY)
#DIAL_API_HOST=
#DIAL_API_VERSION=

# Server Configuration
DEFAULT_MODEL=auto
LOG_LEVEL=INFO
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
DEFAULT_THINKING_MODE_THINKDEEP=high

# Optional Advanced Settings
#DISABLED_TOOLS=
#MAX_MCP_OUTPUT_TOKENS=
#TZ=UTC
"@
        
        Write-Utf8NoBom -Path ".env" -Content $defaultEnv
        Write-Success "Default .env file created"
        Write-Warning "Please edit .env file with your actual API keys"
    }
    else {
        Write-Success ".env file exists"
    }
    
    # Create logs directory for volume mount
    Initialize-Logging
    
    return $true
}

# Start Docker services
function Start-DockerServices {
    param([switch]$Follow = $false)
    
    Write-Step "Starting Docker Services"
    
    # Check if docker-compose.yml exists
    if (!(Test-Path "docker-compose.yml")) {
        Write-Error "docker-compose.yml not found in current directory"
        return $false
    }
    
    try {
        # Stop any existing services
        Write-Info "Stopping any existing services..."
        if (Test-Command "docker-compose") {
            docker-compose down 2>$null
        }
        else {
            docker compose down 2>$null
        }
        
        # Start services
        Write-Info "Starting Unison MCP Server with Docker Compose..."
        if (Test-Command "docker-compose") {
            if ($Follow) {
                docker-compose up --build
            }
            else {
                docker-compose up -d --build
            }
        }
        else {
            if ($Follow) {
                docker compose up --build
            }
            else {
                docker compose up -d --build
            }
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start Docker services"
        }
        
        if (!$Follow) {
            Write-Success "Docker services started successfully"
            Write-Info "Container name: unison-mcp-server"
            Write-Host ""
            Write-Host "To view logs: " -NoNewline
            Write-Host "docker logs -f unison-mcp-server" -ForegroundColor Yellow
            Write-Host "To stop: " -NoNewline
            Write-Host "docker-compose down" -ForegroundColor Yellow
        }
        
        return $true
    }
    catch {
        Write-Error "Failed to start Docker services: $_"
        return $false
    }
}

# Get Docker container status
function Get-DockerStatus {
    try {
        $containerStatus = docker ps --filter "name=unison-mcp-server" --format "{{.Status}}"
        if ($containerStatus) {
            Write-Success "Container status: $containerStatus"
            return $true
        }
        else {
            Write-Warning "Container not running"
            return $false
        }
    }
    catch {
        Write-Warning "Could not get container status: $_"
        return $false
    }
}

# ============================================================================
# End Docker Functions
# ============================================================================

# Setup logging directory
function Initialize-Logging {
    Write-Step "Setting up Logging"
    
    if (!(Test-Path $LOG_DIR)) {
        New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
        Write-Success "Logs directory created"
    }
    else {
        Write-Success "Logs directory already exists"
    }
}

# Check Docker
function Test-Docker {
    Write-Step "Checking Docker Setup"
    
    if ($SkipDocker) {
        Write-Warning "Skipping Docker checks"
        return
    }
    
    if (Test-Command "docker") {
        try {
            $null = docker version 2>$null
            Write-Success "Docker is installed and running"
            
            if (Test-Command "docker-compose") {
                Write-Success "Docker Compose is available"
            }
            else {
                Write-Warning "Docker Compose not found. Install Docker Desktop for Windows."
            }
        }
        catch {
            Write-Warning "Docker is installed but not running. Please start Docker Desktop."
        }
    }
    else {
        Write-Warning "Docker not found. Install Docker Desktop from https://docker.com"
    }
}

# ----------------------------------------------------------------------------
# MCP Client Configuration System
# ----------------------------------------------------------------------------

# Locate the config file of an MSIX-packaged Claude Desktop, or $null when no
# package is present. Package folder names under %LOCALAPPDATA%\Packages are
# NOT stable across distributions — observed as AnthropicPBC.Claude* (Store)
# and Claude_<publisherhash> (MSIX installer) — so discovery is content-based:
# any package whose LocalCache contains the app's virtualized Roaming\Claude
# userData. Name-based fallbacks cover a package that has never been launched.
function Get-ClaudeDesktopMsixConfigPath {
    $packagesRoot = "$env:LOCALAPPDATA\Packages"
    if (!(Test-Path $packagesRoot)) { return $null }

    $dataDirs = @()
    foreach ($pkg in Get-ChildItem -Path $packagesRoot -Directory -ErrorAction SilentlyContinue) {
        $dataDir = Join-Path $pkg.FullName "LocalCache\Roaming\Claude"
        if (Test-Path $dataDir) { $dataDirs += Get-Item $dataDir }
    }
    if ($dataDirs.Count -gt 0) {
        # If several packages carry Claude data (e.g. a stale Store install
        # next to the current one), prefer the most recently active.
        $live = $dataDirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        return (Join-Path $live.FullName "claude_desktop_config.json")
    }

    foreach ($filter in @("AnthropicPBC.Claude*", "Claude_*")) {
        $pkg = Get-ChildItem -Path $packagesRoot -Directory -Filter $filter -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($pkg) {
            return (Join-Path $pkg.FullName "LocalCache\Roaming\Claude\claude_desktop_config.json")
        }
    }
    return $null
}

function Get-ClaudeDesktopConfigPath {
    # MSIX installs of Claude Desktop virtualize %APPDATA% writes into the
    # package's LocalCache, and once a virtualized copy of the config exists
    # it shadows the real %APPDATA%\Claude file for the packaged app.
    # Classic installs read %APPDATA%\Claude directly.
    $msixFile = Get-ClaudeDesktopMsixConfigPath
    if ($msixFile) {
        # An existing virtualized copy always wins. Otherwise target the
        # virtualized path only when there is no classic data dir a classic
        # install would be reading.
        if ((Test-Path $msixFile) -or -not (Test-Path "$env:APPDATA\Claude")) {
            return $msixFile
        }
    }
    return "$env:APPDATA\Claude\claude_desktop_config.json"
}

# Keep the Store (MSIX) and classic Claude Desktop config locations identical.
# An MSIX package may or may not virtualize %APPDATA% depending on its
# manifest (unvirtualizedResources), so from outside the package there is no
# way to know whether the app reads the real %APPDATA%\Claude file or the
# package's LocalCache copy. Whenever a Store package is present, mirror the
# freshly written config to the other candidate location so the server
# registration is visible regardless of the package's virtualization mode.
function Sync-ClaudeDesktopConfigMirror {
    param([Parameter(Mandatory = $true)][string]$WrittenPath)

    $msixFile = Get-ClaudeDesktopMsixConfigPath
    if (!$msixFile) { return }

    $classicFile = "$env:APPDATA\Claude\claude_desktop_config.json"
    $mirror = if ($WrittenPath -eq $msixFile) { $classicFile } else { $msixFile }

    if ($mirror -eq $msixFile -and (Test-Path $msixFile)) {
        # The virtualized copy is the packaged app's live config; primary
        # selection targets it whenever it exists, so never overwrite it with
        # content merged from the classic side.
        return
    }

    try {
        $mirrorDir = Split-Path $mirror -Parent
        if (!(Test-Path $mirrorDir)) {
            New-Item -ItemType Directory -Path $mirrorDir -Force | Out-Null
        }
        if (Test-Path $mirror) {
            Manage-ConfigBackups -ConfigFilePath $mirror | Out-Null
        }
        Copy-Item -Path $WrittenPath -Destination $mirror -Force
        Write-Host "  Mirrored to:  $mirror" -ForegroundColor Gray
        Write-Host "  (Store and classic installs read different locations; both were updated)" -ForegroundColor Gray
    }
    catch {
        Write-Warning "Could not mirror Claude Desktop config to $mirror : $_"
    }
}

# Centralized MCP client definitions
$script:McpClientDefinitions = @(
    @{
        Name           = "Claude Desktop"
        # Claude Desktop install layouts vary: classic installs keep app data
        # at %APPDATA%\Claude (updater under %LOCALAPPDATA%\AnthropicClaude,
        # some builds use %LOCALAPPDATA%\Claude); MSIX deployments live under
        # %LOCALAPPDATA%\Packages with a distribution-dependent package name —
        # observed as AnthropicPBC.Claude* (Store) and Claude_<publisherhash>
        # (MSIX installer). Detect any of them — never the MCP config file
        # itself, which only exists once MCP has been configured. The writer
        # creates the config file (and its directory) at the path the
        # installed variant actually reads.
        DetectionPaths = @(
            "$env:APPDATA\Claude"
            "$env:LOCALAPPDATA\Claude"
            "$env:LOCALAPPDATA\AnthropicClaude"
            "$env:LOCALAPPDATA\Packages\AnthropicPBC.Claude*"
            "$env:LOCALAPPDATA\Packages\Claude_*"
        )
        DetectionType  = "Path"
        ConfigPath     = (Get-ClaudeDesktopConfigPath)
        ConfigJsonPath = "mcpServers.unison"
    },
    @{
        # VS Code reads MCP servers from a user-profile `mcp.json` with a
        # top-level `servers` key. This entry previously wrote `settings.json`
        # with an `mcp.servers` key, which is the older shape — VS Code Insiders
        # below was already updated and stable was not.
        # https://code.visualstudio.com/docs/agent-customization/mcp-servers
        Name             = "VSCode"
        DetectionCommand = "code"
        DetectionType    = "Command"
        ConfigPath       = "$env:APPDATA\Code\User\mcp.json"
        ConfigJsonPath   = "servers.unison"
        IsVSCode         = $true
    },
    @{
        Name             = "VSCode Insiders"
        DetectionCommand = "code-insiders"
        DetectionType    = "Command"
        ConfigPath       = "$env:APPDATA\Code - Insiders\User\mcp.json"
        ConfigJsonPath   = "servers.unison"
        IsVSCodeInsiders = $true
    },
    @{
        Name             = "Cursor"
        DetectionCommand = "cursor"
        DetectionType    = "Command"
        ConfigPath       = "$env:USERPROFILE\.cursor\mcp.json"
        ConfigJsonPath   = "mcpServers.unison"
    },
    @{
        Name           = "Windsurf"
        DetectionPath  = "$env:USERPROFILE\.codeium\windsurf"
        DetectionType  = "Path"
        ConfigPath     = "$env:USERPROFILE\.codeium\windsurf\mcp_config.json"
        ConfigJsonPath = "mcpServers.unison"
    },
    @{
        Name           = "Trae"
        DetectionPath  = "$env:APPDATA\Trae"
        DetectionType  = "Path"
        ConfigPath     = "$env:APPDATA\Trae\User\mcp.json"
        ConfigJsonPath = "mcpServers.unison"
    },
    # CLI hosts. These don't follow the JSON-config pattern above; each row
    # names its Handler function, which Invoke-McpClientConfiguration
    # dispatches instead of Configure-McpClient. They are listed here so this
    # table enumerates EVERY supported host - comparing coverage with
    # run-server.sh means comparing this table against its MCP_HOST_REGISTRY.
    @{
        Name             = "Claude CLI"
        DetectionCommand = "claude"
        DetectionType    = "Command"
        Handler          = "Test-ClaudeCliIntegration"
    },
    @{
        Name             = "Gemini CLI"
        DetectionPath    = "$env:USERPROFILE\.gemini\settings.json"
        DetectionType    = "Path"
        Handler          = "Test-GeminiCliIntegration"
    },
    @{
        Name             = "Qwen CLI"
        DetectionCommand = "qwen"
        DetectionType    = "Command"
        Handler          = "Test-QwenCliIntegration"
    },
    @{
        Name             = "Codex CLI"
        DetectionCommand = "codex"
        DetectionType    = "Command"
        Handler          = "Test-CodexCliIntegration"
    }
)

# Docker MCP configuration template (legacy, kept for backward compatibility)
$script:DockerMcpConfig = @{
    command = "docker"
    args    = @("exec", "-i", "unison-mcp-server", "python", "server.py")
    type    = "stdio"
}

# Generate Docker MCP configuration using docker run (recommended for all clients)
function Get-DockerMcpConfigRun {
    param([string]$ServerPath)
    
    $scriptDir = Split-Path $ServerPath -Parent
    $envFile = Join-Path $scriptDir ".env"
    
    return @{
        command = "docker"
        args    = @("run", "--rm", "-i", "--env-file", $envFile, "unison-mcp-server:latest", "python", "server.py")
        type    = "stdio"
    }
}

# Generate Python MCP configuration
function Get-PythonMcpConfig {
    param([string]$PythonPath, [string]$ServerPath)
    return @{
        command = $PythonPath
        args    = @($ServerPath)
        type    = "stdio"
    }
}

# Check if client uses mcp.json format with servers structure
function Test-McpJsonFormat {
    param([hashtable]$Client)
    
    $configFileName = Split-Path $Client.ConfigPath -Leaf
    return $configFileName -eq "mcp.json"
}

# Check if a client uses the mcp.json format (top-level `servers` key) rather
# than the older `mcpServers` / `mcp.servers` shapes.
#
# Keyed on the configured shape rather than on which client it is, so both
# VS Code stable and Insiders are covered — they now use the same format, and
# tying this to a client flag is what let stable drift onto the old shape.
function Test-VSCodeInsidersFormat {
    param([hashtable]$Client)

    return $Client.ConfigJsonPath -eq "servers.unison"
}

# Analyze existing MCP configuration to determine type (Python or Docker)
function Get-ExistingMcpConfigType {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Client,
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath
    )
    
    if (!(Test-Path $ConfigPath)) {
        return @{
            Exists  = $false
            Type    = "None"
            Details = "No configuration found"
        }
    }
    
    try {
        $content = Get-Content $ConfigPath -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        if (!$content) {
            return @{
                Exists  = $false
                Type    = "None"
                Details = "Invalid JSON configuration"
            }
        }
        
        # Navigate to unison configuration
        $pathParts = $Client.ConfigJsonPath.Split('.')
        $palKey = $pathParts[-1]
        $parentPath = $pathParts[0..($pathParts.Length - 2)]
        
        $targetObject = $content
        foreach ($key in $parentPath) {
            if (!$targetObject.PSObject.Properties[$key]) {
                return @{
                    Exists  = $false
                    Type    = "None"
                    Details = "Configuration structure not found"
                }
            }
            $targetObject = $targetObject.$key
        }
        
        if (!$targetObject.PSObject.Properties[$palKey]) {
            return @{
                Exists  = $false
                Type    = "None"
                Details = "Unison configuration not found"
            }
        }
        
        $palConfig = $targetObject.$palKey
        
        # Analyze configuration type
        if ($palConfig.command -eq "docker") {
            $dockerType = "Unknown"
            $details = "Docker configuration"
            
            if ($palConfig.args -and $palConfig.args.Count -gt 0) {
                if ($palConfig.args[0] -eq "run") {
                    $dockerType = "Docker Run"
                    $details = "Docker run (dedicated container)"
                }
                elseif ($palConfig.args[0] -eq "exec") {
                    $dockerType = "Docker Exec"
                    $details = "Docker exec (existing container)"
                }
                else {
                    $details = "Docker ($($palConfig.args[0]))"
                }
            }
            
            return @{
                Exists  = $true
                Type    = "Docker"
                SubType = $dockerType
                Details = $details
                Command = $palConfig.command
                Args    = $palConfig.args
            }
        }
        elseif ($palConfig.command -and $palConfig.command.EndsWith("python.exe")) {
            $pythonType = "Python"
            $details = "Python virtual environment"
            
            if ($palConfig.command.Contains(".unison_venv")) {
                $details = "Python (unison virtual environment)"
            }
            elseif ($palConfig.command.Contains("venv")) {
                $details = "Python (virtual environment)"
            }
            else {
                $details = "Python (system installation)"
            }
            
            return @{
                Exists  = $true
                Type    = "Python"
                SubType = $pythonType
                Details = $details
                Command = $palConfig.command
                Args    = $palConfig.args
            }
        }
        else {
            return @{
                Exists  = $true
                Type    = "Unknown"
                Details = "Unknown configuration type: $($palConfig.command)"
                Command = $palConfig.command
                Args    = $palConfig.args
            }
        }
        
    }
    catch {
        return @{
            Exists  = $false
            Type    = "Error"
            Details = "Error reading configuration: $_"
        }
    }
}

# Generic MCP client configuration function
function Configure-McpClient {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Client,
        [Parameter(Mandatory = $true)]
        [bool]$UseDocker,
        [string]$PythonPath = "",
        [string]$ServerPath = ""
    )

    Write-Step "Checking $($Client.Name) Integration"

    # Client detection
    $detected = $false
    if ($Client.DetectionType -eq "Command" -and (Test-Command $Client.DetectionCommand)) {
        $detected = $true
    }
    elseif ($Client.DetectionType -eq "Path") {
        $candidatePaths = @()
        if ($Client.DetectionPaths) { $candidatePaths += $Client.DetectionPaths }
        elseif ($Client.DetectionPath) { $candidatePaths += $Client.DetectionPath }
        foreach ($candidate in $candidatePaths) {
            if ($candidate -and (Test-Path ($candidate -as [string]))) {
                $detected = $true
                break
            }
        }
    }

    if (!$detected) {
        Write-Info "$($Client.Name) not detected - skipping integration"
        return
    }
    Write-Info "Found $($Client.Name)"

    # Handle VSCode special logic for profiles
    $configPath = $Client.ConfigPath
    if ($Client.IsVSCode) {
        $userPath = Split-Path $configPath -Parent
        if (!(Test-Path $userPath)) {
            Write-Warning "$($Client.Name) user directory not found. Skipping."
            return
        }
        
        # Find most recent settings.json (default or profile)
        $settingsFiles = @()
        $defaultSettings = $configPath
        if (Test-Path $defaultSettings) {
            $settingsFiles += @{
                Path         = $defaultSettings
                LastModified = (Get-Item $defaultSettings).LastWriteTime
            }
        }
        
        $profilesPath = Join-Path $userPath "profiles"
        if (Test-Path $profilesPath) {
            Get-ChildItem $profilesPath -Directory | ForEach-Object {
                $profileSettings = Join-Path $_.FullName "settings.json"
                if (Test-Path $profileSettings) {
                    $settingsFiles += @{
                        Path         = $profileSettings
                        LastModified = (Get-Item $profileSettings).LastWriteTime
                    }
                }
            }
        }
        
        if ($settingsFiles.Count -gt 0) {
            $configPath = ($settingsFiles | Sort-Object LastModified -Descending | Select-Object -First 1).Path
        }
    }

    # Handle VSCode Insiders special logic for profiles (uses mcp.json)
    if ($Client.IsVSCodeInsiders) {
        $userPath = Split-Path $configPath -Parent
        if (!(Test-Path $userPath)) {
            Write-Warning "$($Client.Name) user directory not found. Skipping."
            return
        }
        
        # Find most recent mcp.json (default or profile)
        $mcpFiles = @()
        $defaultMcp = $configPath
        if (Test-Path $defaultMcp) {
            $mcpFiles += @{
                Path         = $defaultMcp
                LastModified = (Get-Item $defaultMcp).LastWriteTime
            }
        }
        
        $profilesPath = Join-Path $userPath "profiles"
        if (Test-Path $profilesPath) {
            Get-ChildItem $profilesPath -Directory | ForEach-Object {
                $profileMcp = Join-Path $_.FullName "mcp.json"
                if (Test-Path $profileMcp) {
                    $mcpFiles += @{
                        Path         = $profileMcp
                        LastModified = (Get-Item $profileMcp).LastWriteTime
                    }
                }
            }
        }
        
        if ($mcpFiles.Count -gt 0) {
            $configPath = ($mcpFiles | Sort-Object LastModified -Descending | Select-Object -First 1).Path
        }
    }

    # Check if already configured and analyze existing configuration
    $existingConfig = Get-ExistingMcpConfigType -Client $Client -ConfigPath $configPath
    $newConfigType = if ($UseDocker) { "Docker" } else { "Python" }

    # Earlier versions of this script wrote configs with PowerShell 5.1's
    # `-Encoding UTF8`, which prepends a UTF-8 BOM that Node-based hosts fail
    # to JSON.parse — the server silently never appears. Repair such files
    # without prompting: content is preserved, only the encoding changes.
    $hasBom = $false
    if (Test-Path $configPath) {
        $rawBytes = [System.IO.File]::ReadAllBytes($configPath)
        $hasBom = ($rawBytes.Length -ge 3 -and $rawBytes[0] -eq 0xEF -and $rawBytes[1] -eq 0xBB -and $rawBytes[2] -eq 0xBF)
    }

    if ($existingConfig.Exists -and $hasBom) {
        Write-Warning "Existing config has a UTF-8 byte-order mark that breaks $($Client.Name)'s JSON parser - rewriting it without one"
    }
    elseif ($existingConfig.Exists) {
        Write-Info "Found existing Unison MCP configuration in $($Client.Name)"
        Write-Info "  Current: $($existingConfig.Details)"
        Write-Info "  New: $newConfigType configuration"

        if ($existingConfig.Type -eq $newConfigType) {
            Write-Warning "Same configuration type ($($existingConfig.Type)) already exists"
            $response = Read-Host "`nOverwrite existing $($existingConfig.Type) configuration? (y/N)"
        }
        else {
            Write-Warning "Different configuration type detected"
            Write-Info "  Replacing: $($existingConfig.Type) → $newConfigType"
            $response = Read-Host "`nReplace $($existingConfig.Type) with $newConfigType configuration? (y/N)"
        }
        
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Info "Keeping existing configuration in $($Client.Name)"
            return
        }
        
        Write-Info "Proceeding with configuration update..."
    }
    else {
        # User confirmation for new installation
        $response = Read-Host "`nConfigure Unison MCP for $($Client.Name) (mode: $newConfigType)? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Info "Skipping $($Client.Name) integration"
            return
        }
    }

    try {
        # Create config directory if needed
        $configDir = Split-Path $configPath -Parent
        if (!(Test-Path $configDir)) {
            New-Item -ItemType Directory -Path $configDir -Force | Out-Null
        }

        # Backup existing config
        if (Test-Path $configPath) {
            Manage-ConfigBackups -ConfigFilePath $configPath
        }

        # Read or create config
        $config = New-Object PSObject
        $usesMcpJsonFormat = Test-McpJsonFormat -Client $Client
        $usesVSCodeInsidersFormat = Test-VSCodeInsidersFormat -Client $Client
        
        if (Test-Path $configPath) {
            $fileContent = Get-Content $configPath -Raw
            if ($fileContent.Trim()) {
                $config = $fileContent | ConvertFrom-Json -ErrorAction SilentlyContinue
            }
            if ($null -eq $config) { $config = New-Object PSObject }
        }
        
        # Initialize structure for mcp.json format files if they don't exist or are empty
        if ($usesMcpJsonFormat) {
            if ($usesVSCodeInsidersFormat) {
                # For VS Code Insiders format: {"servers": {...}}
                if (!$config.PSObject.Properties["servers"]) {
                    $config | Add-Member -MemberType NoteProperty -Name "servers" -Value (New-Object PSObject)
                }
            }
            else {
                # For other clients format: {"mcpServers": {...}}
                if (!$config.PSObject.Properties["mcpServers"]) {
                    $config | Add-Member -MemberType NoteProperty -Name "mcpServers" -Value (New-Object PSObject)
                }
            }
        }
        
        # Initialize MCP structure for VS Code settings.json if it doesn't exist
        if ($Client.IsVSCode -and $Client.ConfigJsonPath.StartsWith("mcp.")) {
            if (!$config.PSObject.Properties["mcp"]) {
                $config | Add-Member -MemberType NoteProperty -Name "mcp" -Value (New-Object PSObject)
            }
            if (!$config.mcp.PSObject.Properties["servers"]) {
                $config.mcp | Add-Member -MemberType NoteProperty -Name "servers" -Value (New-Object PSObject)
            }
        }

        # Generate server config
        $serverConfig = if ($UseDocker) { 
            # Use docker run for all clients (more reliable than docker exec)
            Get-DockerMcpConfigRun $ServerPath
        }
        else { 
            Get-PythonMcpConfig $PythonPath $ServerPath 
        }

        # Navigate and set configuration
        $pathParts = $Client.ConfigJsonPath.Split('.')
        $palKey = $pathParts[-1]
        $parentPath = $pathParts[0..($pathParts.Length - 2)]
        
        $targetObject = $config
        foreach ($key in $parentPath) {
            if (!$targetObject.PSObject.Properties[$key]) {
                $targetObject | Add-Member -MemberType NoteProperty -Name $key -Value (New-Object PSObject)
            }
            $targetObject = $targetObject.$key
        }

        # Remove legacy zen entries to avoid duplicate or broken MCP servers
        $legacyRemoved = Remove-LegacyServerKeys $targetObject
        if ($legacyRemoved) {
            Write-Info "Removed legacy MCP entries (zen → unison)"
        }

        $targetObject | Add-Member -MemberType NoteProperty -Name $palKey -Value $serverConfig -Force

        # Write config (UTF-8 without BOM — a BOM breaks the host's JSON.parse)
        Write-Utf8NoBom -Path $configPath -Content ($config | ConvertTo-Json -Depth 10)

        # Read back through the JSON parser so an unparseable write is caught
        # here instead of surfacing as the server silently missing in the host.
        $verify = Get-Content $configPath -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($null -eq $verify) {
            Write-Warning "Written config failed JSON validation - please inspect $configPath"
        }

        Write-Success "Successfully configured $($Client.Name)"
        Write-Host "  Config: $configPath" -ForegroundColor Gray
        if ($Client.Name -eq "Claude Desktop") {
            Sync-ClaudeDesktopConfigMirror -WrittenPath $configPath
            Write-Host "  Fully quit Claude Desktop (system tray icon -> Quit; closing the window is not enough), then relaunch" -ForegroundColor Gray
            Write-Host "  The server appears under Settings -> Developer, and via the tools icon in chat" -ForegroundColor Gray
        }
        else {
            Write-Host "  Restart $($Client.Name) to use the new MCP server" -ForegroundColor Gray
        }

    }
    catch {
        Write-Error "Failed to update $($Client.Name) configuration: $_"
    }
}

# Main MCP client configuration orchestrator
function Invoke-McpClientConfiguration {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$UseDocker,
        [string]$PythonPath = "",
        [string]$ServerPath = ""
    )
    
    Write-Step "Checking Client Integrations"

    # One pass over the full host table. JSON-config hosts go through
    # Configure-McpClient; rows that declare a Handler are CLI hosts and
    # dispatch to their handler function (all handlers share the signature
    # PythonPath, ServerPath). CLI handlers are skipped under Docker, matching
    # the previous behavior.
    foreach ($client in $script:McpClientDefinitions) {
        if ($client.Handler) {
            if (!$UseDocker) {
                & $client.Handler $PythonPath $ServerPath
            }
            continue
        }
        Configure-McpClient -Client $client -UseDocker $UseDocker -PythonPath $PythonPath -ServerPath $ServerPath
    }
}

# Keep existing CLI integration functions
function Test-ClaudeCliIntegration {
    param([string]$PythonPath, [string]$ServerPath)
    
    if (!(Test-Command "claude")) {
        return
    }
    
    Write-Info "Claude CLI detected - checking configuration..."

    foreach ($legacy in $script:LegacyServerNames) {
        try { claude mcp remove -s user $legacy 2>$null | Out-Null } catch {}
    }
    
    try {
        $claudeConfig = claude mcp list 2>$null
        if ($claudeConfig -match "unison") {
            Write-Success "Claude CLI already configured for unison server"
            return
        }

        # Perform the registration rather than printing it for the user to run.
        # run-server.sh registers automatically, and setup that leaves an
        # unexecuted command on screen is not equivalent to setup that works.
        Write-Info "Registering unison server with Claude CLI..."
        claude mcp add -s user unison $PythonPath $ServerPath 2>$null | Out-Null

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Registered unison server with Claude CLI"
        }
        else {
            Write-Warning "Automatic registration failed. To configure manually, run:"
            Write-Host "  claude mcp add -s user unison $PythonPath $ServerPath" -ForegroundColor Cyan
        }
    }
    catch {
        Write-Warning "Could not query or configure Claude CLI. To configure manually, run:"
        Write-Host "  claude mcp add -s user unison $PythonPath $ServerPath" -ForegroundColor Cyan
    }
}

function Test-GeminiCliIntegration {
    # Uniform registry-handler signature (PythonPath, ServerPath); this handler
    # only needs the script directory, derived from ServerPath.
    param([string]$PythonPath, [string]$ServerPath)

    $ScriptDir = Split-Path $ServerPath -Parent
    $palWrapper = Join-Path $ScriptDir "unison-mcp-server.cmd"
    
    # Check if Gemini settings file exists (Windows path)
    $geminiConfig = "$env:USERPROFILE\.gemini\settings.json"
    if (!(Test-Path $geminiConfig)) {
        return
    }

    # Load existing config
    $config = @{}
    $configContent = Get-Content $geminiConfig -Raw -ErrorAction SilentlyContinue
    if ($configContent) {
        try { $config = $configContent | ConvertFrom-Json -ErrorAction Stop } catch { $config = @{} }
    }
    if ($null -eq $config -or $config -isnot [System.Collections.IDictionary]) {
        $config = @{}
    }

    if (-not $config.mcpServers -or $config.mcpServers -isnot [System.Collections.IDictionary]) {
        $config.mcpServers = [ordered]@{}
    }

    $legacyRemoved = Remove-LegacyServerKeys $config.mcpServers
    $palConfig = $config.mcpServers.unison
    $needsWrite = $legacyRemoved

    if ($palConfig) {
        if ($palConfig.command -ne $palWrapper) {
            $palConfig.command = $palWrapper
            $needsWrite = $true
        }

        if (!(Test-Path $palWrapper)) {
            Write-Info "Creating wrapper script for Gemini CLI..."
            @"
@echo off
cd /d "%~dp0"
if exist ".unison_venv\Scripts\python.exe" (
    .unison_venv\Scripts\python.exe server.py %*
) else (
    python server.py %*
)
"@ | Out-File -FilePath $palWrapper -Encoding ASCII
            Write-Success "Created unison-mcp-server.cmd wrapper script"
        }

        if ($needsWrite) {
            Manage-ConfigBackups -ConfigFilePath $geminiConfig | Out-Null
            Write-Utf8NoBom -Path $geminiConfig -Content ($config | ConvertTo-Json -Depth 10)
            Write-Success "Updated Gemini CLI configuration (cleaned legacy entries)"
            Write-Host "  Config: $geminiConfig" -ForegroundColor Gray
            Write-Host "  Restart Gemini CLI to use Unison MCP Server" -ForegroundColor Gray
        }
        return
    }

    # Ask user if they want to add Unison to Gemini CLI
    Write-Host ""
    $response = Read-Host "Configure Unison for Gemini CLI? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Info "Skipping Gemini CLI integration"
        return
    }
    
    # Ensure wrapper script exists
    if (!(Test-Path $palWrapper)) {
        Write-Info "Creating wrapper script for Gemini CLI..."
        @"
@echo off
cd /d "%~dp0"
if exist ".unison_venv\Scripts\python.exe" (
    .unison_venv\Scripts\python.exe server.py %*
) else (
    python server.py %*
)
"@ | Out-File -FilePath $palWrapper -Encoding ASCII
        
        Write-Success "Created unison-mcp-server.cmd wrapper script"
    }
    
    # Update Gemini settings
    Write-Info "Updating Gemini CLI configuration..."
    
    try {
        # Create backup with retention management
        $backupPath = Manage-ConfigBackups $geminiConfig
        
        # Ensure mcpServers exists
        if (-not $config.mcpServers -or $config.mcpServers -isnot [System.Collections.IDictionary]) {
            $config.mcpServers = [ordered]@{}
        }
        
        # Add unison server
        $palConfig = @{
            command = $palWrapper
        }
        
        $config.mcpServers | Add-Member -MemberType NoteProperty -Name "unison" -Value $palConfig -Force
        
        # Write updated config
        Write-Utf8NoBom -Path $geminiConfig -Content ($config | ConvertTo-Json -Depth 10)

        Write-Success "Successfully configured Gemini CLI"
        Write-Host "  Config: $geminiConfig" -ForegroundColor Gray
        Write-Host "  Restart Gemini CLI to use Unison MCP Server" -ForegroundColor Gray
        
    }
    catch {
        Write-Error "Failed to update Gemini CLI config: $_"
        Write-Host ""
        Write-Host "Manual config location: $geminiConfig"
        Write-Host "Add this configuration:"
        Write-Host @"
{
  "mcpServers": {
    "unison": {
      "command": "$palWrapper"
    }
  }
}
"@ -ForegroundColor Yellow
    }
}   

function Show-QwenManualConfig {
    param(
        [string]$PythonPath,
        [string]$ServerPath,
        [string]$ScriptDir,
        [string]$ConfigPath,
        [System.Collections.IDictionary]$EnvironmentMap
    )

    Write-Host "Manual config location: $ConfigPath" -ForegroundColor Yellow
    Write-Host "Add or update this entry:" -ForegroundColor Yellow

    if ($EnvironmentMap -and $EnvironmentMap.Count -gt 0) {
        $pairs = $EnvironmentMap.GetEnumerator() | ForEach-Object {
            $escaped = ($_.Value -replace '\\', '\\\\' -replace '"', '\\"')
            '        "{0}": "{1}"' -f $_.Key, $escaped
        }

        Write-Host "{" -ForegroundColor Yellow
        Write-Host "  \"mcpServers\": {" -ForegroundColor Yellow
        Write-Host "    \"unison\": {" -ForegroundColor Yellow
        Write-Host "      \"command\": \"$PythonPath\"," -ForegroundColor Yellow
        Write-Host "      \"args\": [\"$ServerPath\"]," -ForegroundColor Yellow
        Write-Host "      \"cwd\": \"$ScriptDir\"," -ForegroundColor Yellow
        Write-Host "      \"env\": {" -ForegroundColor Yellow
        Write-Host ($pairs -join "`n") -ForegroundColor Yellow
        Write-Host "      }" -ForegroundColor Yellow
        Write-Host "    }" -ForegroundColor Yellow
        Write-Host "  }" -ForegroundColor Yellow
        Write-Host "}" -ForegroundColor Yellow
    }
    else {
        Write-Host "{" -ForegroundColor Yellow
        Write-Host "  \"mcpServers\": {" -ForegroundColor Yellow
        Write-Host "    \"unison\": {" -ForegroundColor Yellow
        Write-Host "      \"command\": \"$PythonPath\"," -ForegroundColor Yellow
        Write-Host "      \"args\": [\"$ServerPath\"]," -ForegroundColor Yellow
        Write-Host "      \"cwd\": \"$ScriptDir\"" -ForegroundColor Yellow
        Write-Host "    }" -ForegroundColor Yellow
        Write-Host "  }" -ForegroundColor Yellow
        Write-Host "}" -ForegroundColor Yellow
    }
}

function Test-QwenCliIntegration {
    param([string]$PythonPath, [string]$ServerPath)

    if (!(Test-Command "qwen")) {
        return
    }

    Write-Info "Qwen CLI detected - checking configuration..."

    $configPath = Join-Path $env:USERPROFILE ".qwen\settings.json"
    $configDir = Split-Path $configPath -Parent
    $scriptDir = Split-Path $ServerPath -Parent

    $configStatus = "missing"
    $legacyRemoved = $false
    $skipPrompt = $false
    $config = @{}

    if (Test-Path $configPath) {
        try {
            Add-Type -AssemblyName System.Web.Extensions -ErrorAction SilentlyContinue
            $serializer = New-Object System.Web.Script.Serialization.JavaScriptSerializer
            $serializer.MaxJsonLength = 67108864
            $rawJson = Get-Content $configPath -Raw
            $config = $serializer.DeserializeObject($rawJson)
            if (-not ($config -is [System.Collections.IDictionary])) {
                $config = @{}
            }

            if ($config.ContainsKey('mcpServers') -and $config['mcpServers'] -is [System.Collections.IDictionary]) {
                $servers = $config['mcpServers']
                $legacyRemoved = (Remove-LegacyServerKeys $servers) -or $legacyRemoved
                if ($servers.Contains('unison') -and $servers['unison'] -is [System.Collections.IDictionary]) {
                    $palConfig = $servers['unison']
                    $commandMatches = ($palConfig['command'] -eq $PythonPath)

                    $argsValue = $palConfig['args']
                    $argsList = @()
                    if ($argsValue -is [System.Collections.IEnumerable] -and $argsValue -isnot [string]) {
                        $argsList = @($argsValue)
                    }
                    elseif ($null -ne $argsValue) {
                        $argsList = @($argsValue)
                    }
                    $argsMatches = ($argsList.Count -eq 1 -and $argsList[0] -eq $ServerPath)

                    $cwdValue = $null
                    if ($palConfig.Contains('cwd')) {
                        $cwdValue = $palConfig['cwd']
                    }
                    $cwdMatches = ([string]::IsNullOrEmpty($cwdValue) -or $cwdValue -eq $scriptDir)

                    if ($commandMatches -and $argsMatches -and $cwdMatches) {
                        $configStatus = $legacyRemoved ? "cleanup" : "match"
                    }
                    else {
                        $configStatus = "mismatch"
                        Write-Warning "Existing Qwen CLI configuration differs from the current setup."
                    }
                }
            }
        }
        catch {
            $configStatus = "invalid"
            Write-Warning "Unable to parse Qwen CLI settings at $configPath ($_)."
            $config = @{}
        }
    }

    $envMap = [ordered]@{}
    if (Test-Path ".env") {
        foreach ($line in Get-Content ".env") {
            $trimmed = $line.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith('#')) {
                continue
            }

            if ($line -match '^\s*([^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2]
                $value = ($value -replace '\s+#.*$', '').Trim()
                if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
                if ([string]::IsNullOrWhiteSpace($value)) {
                    $value = [Environment]::GetEnvironmentVariable($key, "Process")
                }
                if (![string]::IsNullOrWhiteSpace($value) -and $value -notmatch '^your_.*_here$') {
                    $envMap[$key] = $value
                }
            }
        }
    }

    $extraKeys = @(
        "GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "DIAL_API_KEY", "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_URL", "ANTHROPIC_ALLOWED_MODELS", "ANTHROPIC_MODELS_CONFIG_PATH",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ALLOWED_MODELS", "AZURE_MODELS_CONFIG_PATH",
        "CUSTOM_API_URL", "CUSTOM_API_KEY", "CUSTOM_MODEL_NAME", "DEFAULT_MODEL", "GOOGLE_ALLOWED_MODELS",
        "OPENAI_ALLOWED_MODELS", "OPENROUTER_ALLOWED_MODELS", "XAI_ALLOWED_MODELS", "DEFAULT_THINKING_MODE_THINKDEEP",
        "DISABLED_TOOLS", "CONVERSATION_TIMEOUT_HOURS", "MAX_CONVERSATION_TURNS", "LOG_LEVEL", "UNISON_MCP_FORCE_ENV_OVERRIDE"
    )

    foreach ($key in $extraKeys) {
        if (-not $envMap.Contains($key)) {
            $value = [Environment]::GetEnvironmentVariable($key, "Process")
            if (![string]::IsNullOrWhiteSpace($value) -and $value -notmatch '^your_.*_here$') {
                $envMap[$key] = $value
            }
        }
    }

    if ($configStatus -eq "match") {
        Write-Success "Qwen CLI already configured for unison server"
        return
    }

    if ($configStatus -eq "cleanup") {
        Write-Info "Removing legacy Qwen MCP entries from previous zen configuration..."
        $skipPrompt = $true
    }

    $prompt = "Configure Unison for Qwen CLI? (y/N)"
    if ($configStatus -eq "cleanup") {
        $prompt = "Remove legacy Qwen MCP entries and refresh configuration? (Y/n)"
    }
    elseif ($configStatus -eq "mismatch" -or $configStatus -eq "invalid") {
        $prompt = "Update Qwen CLI unison configuration? (y/N)"
    }

    if (-not $skipPrompt) {
        $response = Read-Host $prompt
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Info "Skipping Qwen CLI integration"
            Show-QwenManualConfig $PythonPath $ServerPath $scriptDir $configPath $envMap
            return
        }
    }

    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    if ((Test-Path $configPath) -and $configStatus -ne "missing") {
        Manage-ConfigBackups $configPath | Out-Null
    }

    try {
        if (-not ($config -is [System.Collections.IDictionary])) {
            $config = @{}
        }

        if (-not $config.ContainsKey('mcpServers') -or $config['mcpServers'] -isnot [System.Collections.IDictionary]) {
            $config['mcpServers'] = @{}
        }

        $palConfig = [ordered]@{
            command = $PythonPath
            args    = @($ServerPath)
            cwd     = $scriptDir
        }

        if ($envMap.Count -gt 0) {
            $palConfig['env'] = $envMap
        }

        $config['mcpServers']['unison'] = $palConfig

        $json = ($config | ConvertTo-Json -Depth 20)
        Write-Utf8NoBom -Path $configPath -Content $json

        Write-Success "Successfully configured Qwen CLI"
        Write-Host "  Config: $configPath" -ForegroundColor Gray
        Write-Host "  Restart Qwen CLI to use Unison MCP Server" -ForegroundColor Gray
    }
    catch {
        Write-Error "Failed to update Qwen CLI configuration: $_"
        Show-QwenManualConfig $PythonPath $ServerPath $scriptDir $configPath $envMap
    }
}


# ----------------------------------------------------------------------------
# End MCP Client Configuration System
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# User Interface Functions
# ----------------------------------------------------------------------------

# Show script help
function Test-CodexCliIntegration {
    # Windows counterpart of run-server.sh's check_codex_cli_integration:
    # cleans legacy [mcp_servers.<legacy>] sections from ~/.codex/config.toml,
    # recognises an existing unison entry, and otherwise appends one. The
    # server entry uses the resolved interpreter + server.py (the same shape
    # Get-PythonMcpConfig registers everywhere else on Windows) rather than the
    # bash/uvx launcher the Unix script writes, which has no Windows equivalent.
    param([string]$PythonPath, [string]$ServerPath)

    if (!(Test-Command "codex")) {
        return
    }

    Write-Info "Codex CLI detected - checking configuration..."

    $configPath = Join-Path (Join-Path $env:USERPROFILE ".codex") "config.toml"

    # TOML basic strings treat backslash as an escape, so Windows paths must be
    # escaped before being written.
    $escapedCommand = $PythonPath -replace '\\', '\\' -replace '"', '\"'
    $escapedArg = $ServerPath -replace '\\', '\\' -replace '"', '\"'

    if (Test-Path $configPath) {
        # Remove legacy [mcp_servers.<legacy>] sections (and their subsections),
        # mirroring the line-filter the Unix script applies.
        $lines = Get-Content $configPath
        $output = New-Object System.Collections.Generic.List[string]
        $skip = $false
        $removed = $false

        foreach ($line in $lines) {
            if ($line -match '^\s*\[([^\]]+)\]') {
                $header = $matches[1].Trim()
                $parts = $header.Split('.')
                $isLegacy = $false
                if ($parts.Count -ge 2 -and $parts[0] -eq 'mcp_servers') {
                    $sectionKey = ($parts | Select-Object -Skip 1) -join '.'
                    foreach ($name in $script:LegacyServerNames) {
                        if ($sectionKey -eq $name -or $sectionKey.StartsWith("$name.")) {
                            $isLegacy = $true
                            break
                        }
                    }
                }
                $skip = $isLegacy
                if ($isLegacy) { $removed = $true; continue }
            }
            if (!$skip) { $output.Add($line) }
        }

        if ($removed) {
            Set-Content -Path $configPath -Value ($output -join "`n").TrimEnd()
            Write-Success "Removed legacy Codex MCP entries"
        }

        if (Select-String -Path $configPath -Pattern '\[mcp_servers\.unison\]' -Quiet) {
            Write-Success "Codex CLI already configured for unison server"
            return
        }
    }

    $response = Read-Host "`nConfigure Unison for Codex CLI? (y/N)"
    if ($response -notmatch '^[Yy]') {
        Write-Info "Skipping Codex CLI integration"
        return
    }

    $configDir = Split-Path $configPath -Parent
    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    if (Test-Path $configPath) {
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        Copy-Item $configPath "$configPath.backup_$timestamp"
    }

    $block = @(
        ""
        "[mcp_servers.unison]"
        "command = `"$escapedCommand`""
        "args = [`"$escapedArg`"]"
        "tool_timeout_sec = 1200"
    )

    # Mirror the Unix script's env section, populated from .env when present.
    $envLines = @()
    if (Test-Path ".env") {
        foreach ($line in Get-Content ".env") {
            $trimmed = $line.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith('#')) { continue }
            if ($trimmed -match '^([^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim() -replace '^["'']|["'']$', ''
                if ([string]::IsNullOrWhiteSpace($value) -or $value -match '^your_.*_here$') { continue }
                $escapedValue = $value -replace '\\', '\\' -replace '"', '\"'
                $envLines += "$key = `"$escapedValue`""
            }
        }
    }
    if ($envLines.Count -gt 0) {
        $block += ""
        $block += "[mcp_servers.unison.env]"
        $block += $envLines
    }

    Add-Content -Path $configPath -Value ($block -join "`n")

    Write-Success "Successfully configured Codex CLI"
    Write-Host "  Config: $configPath"
    Write-Host "  Restart Codex CLI to use Unison MCP Server"
}

function Show-Help {
    Write-Host @"
Unison MCP Server - Setup and Launch Script

USAGE:
.\run-server.ps1 [OPTIONS]

OPTIONS:
-Help                   Show this help message
-Version                Show version information
-Follow                 Follow server logs in real time
-Config                 Show configuration instructions for MCP clients
-ClearCache             Clear Python cache files and exit
-Force                  Force recreation of Python virtual environment
-Dev                    Install development dependencies from requirements-dev.txt
-Docker                 Use Docker instead of Python virtual environment
-SkipVenv              Skip Python virtual environment creation
-SkipDocker            Skip Docker checks and cleanup

EXAMPLES:
.\run-server.ps1                      # Normal startup
.\run-server.ps1 -Follow              # Start and follow logs
.\run-server.ps1 -Config              # Show configuration help
.\run-server.ps1 -Dev                 # Include development dependencies
.\run-server.ps1 -Docker              # Use Docker deployment
.\run-server.ps1 -Docker -Follow      # Docker with log following

For more information, visit: https://github.com/izzoa/unison-mcp-server
"@ -ForegroundColor White
}

# Show version information
function Show-Version {
    $version = Get-Version
    Write-Host "Unison MCP Server version: $version" -ForegroundColor Green
    Write-Host "PowerShell Setup Script for Windows" -ForegroundColor Cyan
    Write-Host "Author: GiGiDKR (https://github.com/GiGiDKR)" -ForegroundColor Gray
    Write-Host "Project: izzoa/unison-mcp-server" -ForegroundColor Gray
}

# Show configuration instructions
function Show-ConfigInstructions {
    param(
        [string]$PythonPath = "",
        [string]$ServerPath = "",
        [switch]$UseDocker = $false
    )
    
    Write-Step "Configuration Instructions"
    
    if ($UseDocker) {
        Write-Host "Docker Configuration:" -ForegroundColor Yellow
        Write-Host "The MCP clients have been configured to use Docker containers." -ForegroundColor White
        Write-Host "Make sure the Docker container is running with: docker-compose up -d" -ForegroundColor Cyan
        Write-Host ""
    }
    else {
        Write-Host "Python Virtual Environment Configuration:" -ForegroundColor Yellow
        Write-Host "Python Path: $PythonPath" -ForegroundColor Cyan
        Write-Host "Server Path: $ServerPath" -ForegroundColor Cyan
        Write-Host ""
    }
    
    Write-Host "Supported MCP Clients:" -ForegroundColor Green
    Write-Host "✓ Claude Desktop" -ForegroundColor White
    Write-Host "✓ Claude CLI" -ForegroundColor White  
    Write-Host "✓ VSCode (with MCP extension)" -ForegroundColor White
    Write-Host "✓ VSCode Insiders" -ForegroundColor White
    Write-Host "✓ Cursor" -ForegroundColor White
    Write-Host "✓ Windsurf" -ForegroundColor White
    Write-Host "✓ Trae" -ForegroundColor White
    Write-Host "✓ Gemini CLI" -ForegroundColor White
    Write-Host "✓ Qwen CLI" -ForegroundColor White
    Write-Host ""
    Write-Host "The script automatically detects and configures compatible clients." -ForegroundColor Gray
    Write-Host "Restart your MCP clients after configuration to use the Unison MCP Server." -ForegroundColor Yellow
}

# Show setup instructions
function Show-SetupInstructions {
    param(
        [string]$PythonPath = "",
        [string]$ServerPath = "",
        [switch]$UseDocker = $false
    )
    
    Write-Step "Setup Complete"
    
    if ($UseDocker) {
        Write-Success "Unison MCP Server is configured for Docker deployment"
        Write-Host "Docker command: docker exec -i unison-mcp-server python server.py" -ForegroundColor Cyan
    }
    else {
        Write-Success "Unison MCP Server is configured for Python virtual environment"
        Write-Host "Python: $PythonPath" -ForegroundColor Cyan
        Write-Host "Server: $ServerPath" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "MCP clients will automatically connect to the server." -ForegroundColor Green
    Write-Host "For manual configuration, use the paths shown above." -ForegroundColor Gray
}

# Start the server
function Start-Server {
    Write-Step "Starting Unison MCP Server"
    
    $pythonPath = "$VENV_PATH\Scripts\python.exe"
    if (!(Test-Path $pythonPath)) {
        Write-Error "Python virtual environment not found. Please run setup first."
        return
    }
    
    $serverPath = "server.py"
    if (!(Test-Path $serverPath)) {
        Write-Error "Server script not found: $serverPath"
        return
    }
    
    try {
        Write-Info "Launching server..."
        & $pythonPath $serverPath
    }
    catch {
        Write-Error "Failed to start server: $_"
    }
}

# Follow server logs
function Follow-Logs {
    Write-Step "Following Server Logs"
    
    $logPath = Join-Path $LOG_DIR $LOG_FILE
    
    if (!(Test-Path $logPath)) {
        Write-Warning "Log file not found: $logPath"
        Write-Info "Starting server to generate logs..."
        Start-Server
        return
    }
    
    try {
        Write-Info "Following logs at: $logPath"
        Write-Host "Press Ctrl+C to stop following logs"
        Write-Host ""
        Get-Content $logPath -Wait
    }
    catch {
        Write-Error "Failed to follow logs: $_"
    }
}

# ----------------------------------------------------------------------------
# Environment File Management
# ----------------------------------------------------------------------------

# Initialize .env file if it doesn't exist
function Initialize-EnvFile {
    Write-Step "Setting up Environment File"
    
    if (!(Test-Path ".env")) {
        Write-Info "Creating default .env file..."
        $defaultEnvContent = @"
# API keys — the server enables one provider per REAL value below. ONLY these
# variables activate providers; placeholder values count as unset.
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
DIAL_API_KEY=your_dial_api_key_here

# Azure OpenAI (both required together)
#AZURE_OPENAI_API_KEY=
#AZURE_OPENAI_ENDPOINT=

# Local/self-hosted OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, ...).
# A real URL alone is enough; leave CUSTOM_API_KEY empty for keyless servers.
#CUSTOM_API_URL=http://localhost:11434/v1
#CUSTOM_API_KEY=
#CUSTOM_MODEL_NAME=llama3.2

# DIAL extras (only meaningful alongside a real DIAL_API_KEY)
#DIAL_API_HOST=
#DIAL_API_VERSION=

# Server Configuration
DEFAULT_MODEL=auto
LOG_LEVEL=INFO
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
DEFAULT_THINKING_MODE_THINKDEEP=high

# Optional Advanced Settings
#DISABLED_TOOLS=
#MAX_MCP_OUTPUT_TOKENS=
#TZ=UTC
"@
        Write-Utf8NoBom -Path ".env" -Content $defaultEnvContent

        Write-Success "Default .env file created"
        Write-Warning "Please edit .env file with your actual API keys"
    }
    else {
        Write-Success ".env file already exists"
    }
}

# Import environment variables from .env file
function Import-EnvFile {
    if (!(Test-Path ".env")) {
        Write-Warning "No .env file found"
        return
    }
    
    try {
        $envContent = Get-Content ".env" -ErrorAction Stop
        foreach ($line in $envContent) {
            if ($line -match '^([^#][^=]*?)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim() -replace '^["'']|["'']$', ''
                
                # Set environment variable for the current session
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
            }
        }
        Write-Success "Environment variables loaded from .env file"
    }
    catch {
        Write-Warning "Could not load .env file: $_"
    }
}

# ----------------------------------------------------------------------------
# Workflow Functions
# ----------------------------------------------------------------------------

# Post-install smoke test: the server module must import with the venv
# interpreter before any MCP client is configured or success is reported.
# Handler registration happens at import time, so this catches broken
# installs (missing or incompatible packages) at setup instead of at the
# first client launch. No API key is needed: provider configuration only
# happens in the server's main().
function Test-ServerImport {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    Write-Step "Verifying Server Installation"
    Write-Info "Checking that the server module imports..."
    & $PythonPath -c "import server"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Server import check failed (exit code $LASTEXITCODE) - see the error above."
        Write-Info "MCP clients were NOT configured. Fix the error and run .\run-server.ps1 again."
        return $false
    }
    Write-Success "Server module imports cleanly"
    return $true
}

# Docker deployment workflow
function Invoke-DockerWorkflow {
    Write-Step "Starting Docker Workflow"
    Write-Host "Unison MCP Server" -ForegroundColor Green
    Write-Host "=================" -ForegroundColor Cyan
    
    $version = Get-Version
    Write-Host "Version: $version"
    Write-Host "Mode: Docker Container" -ForegroundColor Yellow
    Write-Host ""
    
    # Docker setup and validation
    if (!(Test-DockerRequirements)) { exit 1 }
    if (!(Initialize-DockerEnvironment)) { exit 1 }
    
    Import-EnvFile
    $null = Test-ApiKeys
    
    if (!(Build-DockerImage -Force:$Force)) { exit 1 }
    
    # Configure MCP clients for Docker
    Invoke-McpClientConfiguration -UseDocker $true
    
    Show-SetupInstructions -UseDocker
    
    # Start Docker services
    Write-Step "Starting Unison MCP Server"
    if ($Follow) {
        Write-Info "Starting server and following logs..."
        Start-DockerServices -Follow
        exit 0
    }
    
    if (!(Start-DockerServices)) { exit 1 }
    
    Write-Host ""
    Write-Success "Unison MCP Server is running in Docker!"
    Write-Host ""
    
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Restart your MCP clients (Claude Desktop, etc.)" -ForegroundColor White
    Write-Host "2. The server is now ready to use" -ForegroundColor White
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  View logs: " -NoNewline -ForegroundColor White
    Write-Host "docker logs -f unison-mcp-server" -ForegroundColor Yellow
    Write-Host "  Stop server: " -NoNewline -ForegroundColor White
    Write-Host "docker-compose down" -ForegroundColor Yellow
    Write-Host "  Restart server: " -NoNewline -ForegroundColor White
    Write-Host "docker-compose restart" -ForegroundColor Yellow
}

# Python virtual environment deployment workflow
function Invoke-PythonWorkflow {
    Write-Step "Starting Python Virtual Environment Workflow"
    Write-Host "Unison MCP Server" -ForegroundColor Green
    Write-Host "=================" -ForegroundColor Cyan
    
    $version = Get-Version
    Write-Host "Version: $version"
    Write-Host ""
    
    if (!(Test-Path $VENV_PATH)) {
        Write-Info "Setting up Python environment for first time..."
    }
    
    # Python environment setup
    Cleanup-Docker
    Clear-PythonCache
    Initialize-EnvFile
    Import-EnvFile
    $null = Test-ApiKeys
    
    try {
        $pythonPath = Initialize-Environment
    }
    catch {
        Write-Error "Failed to setup Python environment: $_"
        exit 1
    }
    
    try {
        Install-Dependencies $pythonPath -InstallDevDependencies:$Dev
    }
    catch {
        Write-Error "Failed to install dependencies: $_"
        exit 1
    }

    if (!(Test-ServerImport -PythonPath $pythonPath)) {
        exit 1
    }

    $serverPath = Get-AbsolutePath "server.py"
    
    # Configure MCP clients for Python
    Invoke-McpClientConfiguration -UseDocker $false -PythonPath $pythonPath -ServerPath $serverPath
    
    Show-SetupInstructions $pythonPath $serverPath
    Initialize-Logging
    
    Write-Host ""
    Write-Host "Logs will be written to: $(Get-AbsolutePath $LOG_DIR)\$LOG_FILE"
    Write-Host ""
    
    if ($Follow) {
        Follow-Logs
    }
    else {
        Write-Host "To follow logs: .\run-server.ps1 -Follow" -ForegroundColor Yellow
        Write-Host "To show config: .\run-server.ps1 -Config" -ForegroundColor Yellow
        Write-Host "To update: git pull, then run .\run-server.ps1 again" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Happy coding! 🎉" -ForegroundColor Green
        
        $response = Read-Host "`nStart the server now? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Start-Server
        }
    }
}

# ----------------------------------------------------------------------------
# End Workflow Functions
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------

# Main execution function
function Start-MainProcess {
    # Parse command line arguments
    if ($Help) {
        Show-Help
        exit 0
    }
    
    if ($Version) {
        Show-Version  
        exit 0
    }
    
    if ($ClearCache) {
        Clear-PythonCache
        Write-Success "Cache cleared successfully"
        Write-Host ""
        Write-Host "You can now run '.\run-server.ps1' normally"
        exit 0
    }
    
    if ($Config) {
        # Setup minimal environment to get paths for config display
        Write-Info "Setting up environment for configuration display..."
        Write-Host ""
        try {
            if ($Docker) {
                # Docker configuration mode
                if (!(Test-DockerRequirements)) {
                    exit 1
                }
                Initialize-DockerEnvironment
                Show-ConfigInstructions "" "" -UseDocker
            }
            else {
                # Python virtual environment configuration mode
                $pythonPath = Initialize-Environment
                $serverPath = Get-AbsolutePath "server.py"
                Show-ConfigInstructions $pythonPath $serverPath
            }
        }
        catch {
            Write-Error "Failed to setup environment for configuration: $_"
            exit 1
        }
        exit 0
    }

    # ============================================================================
    # Docker Workflow
    # ============================================================================
    if ($Docker) {
        Invoke-DockerWorkflow
        exit 0
    }

    # ============================================================================
    # Python Virtual Environment Workflow (Default)
    # ============================================================================
    Invoke-PythonWorkflow
    exit 0
}

# ============================================================================
# Main Script Execution
# ============================================================================

# Execute main process
Start-MainProcess
