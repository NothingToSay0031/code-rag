# code-rag installer for Windows
# Usage:
#   From a cloned repo:  .\install.ps1
#   One-liner (remote):  irm https://raw.githubusercontent.com/NothingToSay0031/code-rag/main/install.ps1 | iex
#
# Environment overrides:
#   $env:CODE_RAG_INSTALL_DIR = "C:\custom\path"   (default: $HOME\.code-rag)
#   $env:CODE_RAG_REPO_URL    = "https://github.com/NothingToSay0031/code-rag"
#   $env:CODE_RAG_NO_CUDA     = "1"                 (force CPU-only install)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

$REPO_URL    = if ($env:CODE_RAG_REPO_URL) { $env:CODE_RAG_REPO_URL } else { "https://github.com/NothingToSay0031/code-rag" }
$INSTALL_DIR = if ($env:CODE_RAG_INSTALL_DIR) { $env:CODE_RAG_INSTALL_DIR } else { "$HOME\.code-rag" }
$BIN_DIR     = "$HOME\.local\bin"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    WARNING: $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "    ERROR: $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  code-rag installer" -ForegroundColor Magenta
Write-Host "  ==================" -ForegroundColor Magenta
Write-Host ""

# ── 1. Install uv if missing ─────────────────────────────────────────────────
Write-Step "Checking uv..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "    uv not found — installing..."
    $uvInstall = (Invoke-RestMethod "https://astral.sh/uv/install.ps1")
    Invoke-Expression $uvInstall
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Err "uv installation failed. Please install uv manually: https://docs.astral.sh/uv/"
    exit 1
}
Write-Ok "uv $(uv --version 2>&1)"

# ── 2. Detect GPU & pick CUDA variant ────────────────────────────────────────
Write-Step "Detecting GPU..."

$CUDA_INDEX_SUFFIX = ""
$GPU_INFO          = "none"

if (-not $env:CODE_RAG_NO_CUDA) {
    # ── 2a. Detect NVIDIA GPU ────────────────────────────────────────────────
    $hasNvidia = $false
    $gpuName   = ""
    try {
        $wmicLines = wmic path win32_videocontroller get name 2>&1
        $nvidiaLine = $wmicLines | Where-Object { $_ -match "NVIDIA" } | Select-Object -First 1
        if ($nvidiaLine) {
            $hasNvidia = $true
            $gpuName   = $nvidiaLine.Trim()
        }
    } catch {
        Write-Warn "wmic GPU detection failed: $_ — trying CimInstance..."
    }

    if (-not $hasNvidia) {
        try {
            $gpuList = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue
            $nvidiaGpu = $gpuList | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
            if ($nvidiaGpu) {
                $hasNvidia = $true
                $gpuName   = $nvidiaGpu.Name.Trim()
            }
        } catch {}
    }

    if ($hasNvidia) {
        Write-Ok "NVIDIA GPU detected: $gpuName"

        # ── 2b. Determine CUDA version ───────────────────────────────────────
        $cudaVer = ""

        # Method 1: nvidia-smi (driver-level, most reliable)
        try {
            $smiPath = $null
            $smiCandidates = @(
                "$env:SystemRoot\System32\nvidia-smi.exe",
                "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
            )
            foreach ($c in $smiCandidates) {
                if (Test-Path $c) { $smiPath = $c; break }
            }
            if (-not $smiPath) {
                $smiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
                if ($smiCmd) { $smiPath = $smiCmd.Source }
            }
            if ($smiPath) {
                $smiOut = & $smiPath 2>&1 | Out-String
                $smiMatch = [regex]::Match($smiOut, 'CUDA Version:\s*([\d.]+)')
                if ($smiMatch.Success) {
                    $cudaVer = $smiMatch.Groups[1].Value
                    Write-Ok "CUDA version from nvidia-smi: $cudaVer"
                }
            }
        } catch {}

        # Method 2: Windows registry
        if (-not $cudaVer) {
            try {
                $regBase = "HKLM:\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
                $cudaKeys = Get-ChildItem $regBase -ErrorAction SilentlyContinue
                if ($cudaKeys) {
                    $latestKey = $cudaKeys |
                        Sort-Object { [version](($_.PSChildName -replace '^v','') + ".0") } -ErrorAction SilentlyContinue |
                        Select-Object -Last 1
                    if ($latestKey) {
                        $cudaVer = $latestKey.PSChildName -replace '^v', ''
                        Write-Ok "CUDA version from registry: $cudaVer"
                    }
                }
            } catch {}
        }

        # Method 3: nvcc
        if (-not $cudaVer) {
            $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
            if ($nvcc) {
                try {
                    $nvccOut  = & nvcc --version 2>&1 | Out-String
                    $verMatch = [regex]::Match($nvccOut, 'release\s+([\d.]+)')
                    if ($verMatch.Success) {
                        $cudaVer = $verMatch.Groups[1].Value
                        Write-Ok "CUDA version from nvcc: $cudaVer"
                    }
                } catch {}
            }
        }

        # ── 2c. Map CUDA version → torch index suffix ────────────────────────
        if ($cudaVer) {
            $parts = $cudaVer.Split(".")
            $major = [int]$parts[0]
            $minor = if ($parts.Length -gt 1) { [int]$parts[1] } else { 0 }

            if     ($major -gt 12 -or ($major -eq 12 -and $minor -ge 8)) { $CUDA_INDEX_SUFFIX = "cu128" }
            elseif ($major -eq 12 -and $minor -ge 4)                     { $CUDA_INDEX_SUFFIX = "cu124" }
            elseif ($major -eq 12 -and $minor -ge 1)                     { $CUDA_INDEX_SUFFIX = "cu121" }
            elseif ($major -ge 11 -and ($major -gt 11 -or $minor -ge 8)) { $CUDA_INDEX_SUFFIX = "cu118" }
            else {
                Write-Warn "CUDA $cudaVer is older than 11.8 — falling back to CPU torch"
            }

            if ($CUDA_INDEX_SUFFIX) {
                $GPU_INFO = "CUDA $cudaVer -> torch index: $CUDA_INDEX_SUFFIX"
            }
        } else {
            Write-Warn "CUDA version not detected — defaulting to cu128 (set CODE_RAG_NO_CUDA=1 to force CPU)"
            $CUDA_INDEX_SUFFIX = "cu128"
            $GPU_INFO = "CUDA version unknown -> torch index: cu128"
        }
    }
}

if ($CUDA_INDEX_SUFFIX) {
    Write-Ok "$GPU_INFO"
} else {
    Write-Ok "No NVIDIA GPU (or CODE_RAG_NO_CUDA set) — will use CPU torch"
}

# ── 3. Clone or update repo ───────────────────────────────────────────────────
Write-Step "Setting up code-rag source..."

$scriptDir = $PSScriptRoot
if ($scriptDir -and (Test-Path (Join-Path $scriptDir "pyproject.toml"))) {
    if ($env:CODE_RAG_INSTALL_DIR -and $env:CODE_RAG_INSTALL_DIR -ne $scriptDir) {
        Write-Warn "CODE_RAG_INSTALL_DIR is set but ignored — running from existing repo ($scriptDir)"
    }
    $INSTALL_DIR = $scriptDir
    Write-Ok "Using existing directory: $INSTALL_DIR"
} elseif (Test-Path (Join-Path $INSTALL_DIR "pyproject.toml")) {
    Write-Host "    Updating existing install in $INSTALL_DIR ..."
    git -C $INSTALL_DIR pull --ff-only
    Write-Ok "Updated"
} else {
    Write-Host "    Cloning from $REPO_URL ..."
    git clone $REPO_URL $INSTALL_DIR
    Write-Ok "Cloned to $INSTALL_DIR"
}

# ── 4. Install base dependencies ─────────────────────────────────────────────
Write-Step "Installing base dependencies..."
Push-Location $INSTALL_DIR
try {
    Write-Host "    Running: uv sync --no-dev"
    uv sync --no-dev
    Write-Ok "Base dependencies installed"
} finally {
    Pop-Location
}

# ── 5. Install torch (CUDA or CPU) ───────────────────────────────────────────
#    This is done AFTER uv sync because sentence-transformers pulls in CPU torch
#    from PyPI. We must uninstall it first, then install the correct CUDA variant
#    from the PyTorch wheel index.
Write-Step "Installing PyTorch..."

$venvPython = Join-Path $INSTALL_DIR ".venv\Scripts\python.exe"

if ($CUDA_INDEX_SUFFIX) {
    $torchIndexUrl = "https://download.pytorch.org/whl/$CUDA_INDEX_SUFFIX"

    # Step 5a: Remove CPU torch that uv sync installed via sentence-transformers
    Write-Host "    Removing CPU torch (pulled in by sentence-transformers)..."
    uv pip uninstall torch torchvision torchaudio 2>$null --python $venvPython
    # Ignore errors if not installed

    # Step 5b: Install CUDA torch from pytorch index
    Write-Host "    Installing CUDA torch from: $torchIndexUrl"
    uv pip install torch torchvision torchaudio --index-url $torchIndexUrl --python $venvPython
    Write-Ok "PyTorch installed (target: $CUDA_INDEX_SUFFIX)"
} else {
    Write-Host "    CPU torch already installed via uv sync — verifying..."
    $hasTorch = & $venvPython -c "import torch; print('ok')" 2>&1
    if ($hasTorch -ne "ok") {
        Write-Host "    Installing CPU torch..."
        uv pip install torch torchvision torchaudio --python $venvPython
    }
    Write-Ok "PyTorch installed (CPU)"
}

# ── 6. Verify PyTorch installation ───────────────────────────────────────────
Write-Step "Verifying PyTorch installation..."

$torchVer = & $venvPython -c "import torch; print(torch.__version__)" 2>&1
Write-Host "    torch version: $torchVer"

if ($CUDA_INDEX_SUFFIX) {
    $cudaAvail  = & $venvPython -c "import torch; print(torch.cuda.is_available())" 2>&1
    $torchCuda  = & $venvPython -c "import torch; print(torch.version.cuda)" 2>&1

    if ($cudaAvail -eq "True") {
        $deviceName = & $venvPython -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
        Write-Ok "torch.cuda.is_available() = True"
        Write-Ok "torch.version.cuda = $torchCuda"
        Write-Ok "GPU device: $deviceName"
    } else {
        Write-Host ""
        Write-Warn "torch.cuda.is_available() = False"
        Write-Warn "torch.version.cuda = $torchCuda"
        Write-Host ""
        Write-Host "    CUDA torch was requested but is not working." -ForegroundColor Yellow
        Write-Host "    Possible causes:" -ForegroundColor Yellow
        Write-Host "      1. NVIDIA driver too old for CUDA $CUDA_INDEX_SUFFIX" -ForegroundColor Yellow
        Write-Host "      2. uv cached the CPU wheel and didn't actually replace it" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    Try this manual fix:" -ForegroundColor Yellow
        Write-Host "      cd $INSTALL_DIR" -ForegroundColor White
        Write-Host "      uv pip uninstall torch --python .venv\Scripts\python.exe" -ForegroundColor White
        Write-Host "      uv cache clean torch" -ForegroundColor White
        Write-Host "      uv pip install torch --index-url https://download.pytorch.org/whl/$CUDA_INDEX_SUFFIX --python .venv\Scripts\python.exe --no-cache" -ForegroundColor White
        Write-Host ""
        Write-Host "    Or try a lower CUDA version:" -ForegroundColor Yellow
        Write-Host "      uv pip uninstall torch --python .venv\Scripts\python.exe" -ForegroundColor White
        Write-Host "      uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --python .venv\Scripts\python.exe --no-cache" -ForegroundColor White
        Write-Host ""
        Write-Host "    Or force CPU mode:" -ForegroundColor Yellow
        Write-Host '      $env:CODE_RAG_NO_CUDA = "1"' -ForegroundColor White
        Write-Host "      .\install.ps1" -ForegroundColor White
    }
}

# ── 7. Create bin wrapper ─────────────────────────────────────────────────────
Write-Step "Creating code-rag wrapper..."
$null = New-Item -ItemType Directory -Force $BIN_DIR
$exe  = Join-Path $INSTALL_DIR ".venv\Scripts\code-rag.exe"
$bat  = Join-Path $BIN_DIR "code-rag.bat"
@"
@echo off
"$exe" %*
"@ | Set-Content $bat -Encoding ASCII
Write-Ok "Wrapper: $bat"

# ── 8. Add BIN_DIR to user PATH if needed ─────────────────────────────────────
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$BIN_DIR*") {
    Write-Step "Adding $BIN_DIR to user PATH..."
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$BIN_DIR", "User")
    $env:PATH = "$env:PATH;$BIN_DIR"
    Write-Ok "PATH updated (restart terminal to take effect)"
}

# ── 9. Done ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor Yellow
Write-Host "    code-rag init C:\path\to\your\project"
Write-Host "    Then open the project in your AI editor -> MCP is auto-configured."
Write-Host ""
Write-Host "  To verify:"
Write-Host "    code-rag --version"
Write-Host ""