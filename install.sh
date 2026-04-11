#!/usr/bin/env bash
# code-rag installer for macOS / Linux
# Usage:
#   From a cloned repo:  bash install.sh
#   One-liner (remote):  curl -sSfL https://raw.githubusercontent.com/NothingToSay0031/code-rag/main/install.sh | bash
#
# Environment overrides:
#   CODE_RAG_INSTALL_DIR=/custom/path   (default: $HOME/.code-rag)
#   CODE_RAG_REPO_URL=https://...       (GitHub URL)
#   CODE_RAG_NO_CUDA=1                  (force CPU-only install)

set -euo pipefail

REPO_URL="${CODE_RAG_REPO_URL:-https://github.com/NothingToSay0031/code-rag}"
INSTALL_DIR="${CODE_RAG_INSTALL_DIR:-$HOME/.code-rag}"
BIN_DIR="$HOME/.local/bin"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; MAGENTA='\033[0;35m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}==> $*${NC}"; }
ok()    { echo -e "    ${GREEN}$*${NC}"; }
warn()  { echo -e "    ${YELLOW}WARNING: $*${NC}"; }

echo ""
echo -e "  ${MAGENTA}code-rag installer${NC}"
echo -e "  ${MAGENTA}==================${NC}"
echo ""

# ── 1. Install uv if missing ──────────────────────────────────────────────────
step "Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "    uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>&1)"

# ── 2. Detect GPU & pick CUDA extra ──────────────────────────────────────────
step "Detecting GPU..."

UV_EXTRA=""

if [[ -z "${CODE_RAG_NO_CUDA:-}" ]]; then
    # Try nvidia-smi first (works on Linux + Windows WSL)
    CUDA_VER=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null \
            | grep -oP 'CUDA Version:\s*\K[\d.]+' \
            | head -1 || true)
    fi
    # Fallback: try nvcc
    if [[ -z "$CUDA_VER" ]] && command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null \
            | grep -oP 'release \K[\d.]+' \
            | head -1 || true)
    fi

    if [[ -n "$CUDA_VER" ]]; then
        MAJOR=${CUDA_VER%%.*}
        MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        MINOR=${MINOR:-0}

        if   (( MAJOR > 12 )) || (( MAJOR == 12 && MINOR >= 8 )); then UV_EXTRA="cu128"
        elif (( MAJOR == 12 && MINOR >= 4 ));                      then UV_EXTRA="cu124"
        elif (( MAJOR == 12 && MINOR >= 1 ));                      then UV_EXTRA="cu121"
        elif (( MAJOR >= 11 )) && (( MAJOR > 11 || MINOR >= 8 ));  then UV_EXTRA="cu118"  # incl. CUDA 12.0
        else
            warn "CUDA $CUDA_VER is older than 11.8 — falling back to CPU torch"
        fi

        if [[ -n "$UV_EXTRA" ]]; then
            ok "GPU detected: CUDA $CUDA_VER -> will install torch ($UV_EXTRA)"
        fi
    fi
fi

if [[ -z "$UV_EXTRA" ]]; then
    ok "No NVIDIA GPU (or CODE_RAG_NO_CUDA set) — using CPU torch"
fi

# ── 3. Clone or update repo ───────────────────────────────────────────────────
step "Setting up code-rag source..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    INSTALL_DIR="$SCRIPT_DIR"
    ok "Using existing directory: $INSTALL_DIR"
elif [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
    echo "    Updating existing install in $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" pull --ff-only
    ok "Updated"
else
    echo "    Cloning from $REPO_URL ..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    ok "Cloned to $INSTALL_DIR"
fi

# ── 4. Install dependencies (one uv sync, right torch version) ───────────────
step "Installing dependencies..."
cd "$INSTALL_DIR"

if [[ -n "$UV_EXTRA" ]]; then
    echo "    Running: uv sync --extra $UV_EXTRA --no-dev"
    uv sync --extra "$UV_EXTRA" --no-dev
    # uv.lock conflict workaround: when PyPI torch and pytorch-index torch share the
    # same base version (e.g. 2.11.0 vs 2.11.0+cu128), uv may install the CPU-only
    # PyPI build despite --extra cuXXX being active.
    # Explicitly reinstall from the pytorch CUDA index to guarantee the right wheel.
    echo "    Ensuring CUDA torch from pytorch index (uv lock workaround)..."
    "$INSTALL_DIR/.venv/bin/python" -m pip install torch \
        --index-url "https://download.pytorch.org/whl/$UV_EXTRA" \
        --upgrade --quiet
    ok "Installed with CUDA torch ($UV_EXTRA)"
else
    echo "    Running: uv sync --no-dev"
    uv sync --no-dev
    ok "Installed with CPU torch"
fi

# ── 5. Create bin wrapper ─────────────────────────────────────────────────────
step "Creating code-rag wrapper..."
mkdir -p "$BIN_DIR"
WRAPPER="$BIN_DIR/code-rag"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/code-rag" "\$@"
EOF
chmod +x "$WRAPPER"
ok "Wrapper: $WRAPPER"

# ── 6. Add BIN_DIR to shell PATH if needed ─────────────────────────────────────
add_to_path() {
    local profile="$1"
    [[ -f "$profile" ]] && grep -q "$BIN_DIR" "$profile" 2>/dev/null && return
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$profile"
    ok "Added to $profile"
}

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    step "Adding $BIN_DIR to shell PATH..."
    case "${SHELL:-}" in
        */zsh)  add_to_path "$HOME/.zshrc" ;;
        */fish)
            mkdir -p "$HOME/.config/fish"
            echo "fish_add_path $BIN_DIR" >> "$HOME/.config/fish/config.fish"
            ok "Added to fish config"
            ;;
        *)
            add_to_path "$HOME/.bashrc"
            add_to_path "$HOME/.bash_profile"
            ;;
    esac
    export PATH="$BIN_DIR:$PATH"
fi

# ── 7. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}Installation complete!${NC}"
echo ""
echo -e "  ${YELLOW}Quick start:${NC}"
echo "    code-rag init /path/to/your/project"
echo "    Then open the project in your AI editor — MCP is auto-configured."
echo ""
echo "  To verify:"
echo "    code-rag --version"
echo ""
